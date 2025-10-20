import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

from pathlib import Path
from typing import List, Dict

class Config:
    DATA_PATH: str = "../../reports/data.csv"
    MODEL_PATH: str = "predictive_model.joblib"
    CLUSTER_MODEL_PATH: str = "clustering_model.joblib"
    PREPROCESSOR_PATH: str = "preprocessor.joblib"
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    N_CLUSTERS: int = 4
    CATEGORICAL_COLUMNS: List[str] = [
        'Gender', 'Income', 'Customer_Segment',
        'Product_Category', 'Shipping_Method', 'Payment_Method'
    ]
    FEATURE_COLUMNS: List[str] = [
        'Year', 'Month', 'Day', 'Weekday', 'Hour', 'Product_Category',
        'Customer_Segment', 'Shipping_Method', 'Payment_Method', 'Amount',
        'Total_Amount', 'Ratings', 'Age', 'Rolling_7d', 'Rolling_14d', 'Segment_Product'
    ]
    CLUSTER_FEATURES: List[str] = ['Age', 'Income', 'Total_Purchases', 'Ratings', 'Amount', 'Total_Amount']
    MODEL_PARAMS: dict = {
        'n_estimators': [100, 150],
        'learning_rate': [0.01, 0.05],
        'max_depth': [3, 5, 7],
        'num_leaves': [15, 31, 63]
    }

class DataPreprocessor:
    def __init__(self):
        self.category_mappings: Dict[str, Dict] = {}
        self.scaler: StandardScaler = None
        self._preprocessed_data = None
        self._X = None
        self._y = None

    def load_and_preprocess(self):
        if self._preprocessed_data is not None:
            return self._X, self._y

        df = pd.read_csv(Config.DATA_PATH)

        for col in df.columns:
            if df[col].dtype in ['int64','float64']:
                df[col] = df[col].fillna(df[col].mean())
            elif col in Config.CATEGORICAL_COLUMNS:
                df[col] = df[col].fillna(df[col].mode()[0])

        df['Time'] = df.get('Time', '00:00:00').replace('######', pd.NaT)
        df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour
        df['Hour'] = df['Hour'].fillna(df['Hour'].mean())

        for col in Config.CATEGORICAL_COLUMNS:
            df[col] = df[col].astype('category')
            self.category_mappings[col] = dict(enumerate(df[col].cat.categories))
            self.category_mappings[f"{col}_inv"] = {v:k for k,v in self.category_mappings[col].items()}
            df[col] = df[col].cat.codes

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Day'] = df['Date'].dt.day
        df['Weekday'] = df['Date'].dt.weekday
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month

        df = df.sort_values('Date')
        df['Rolling_7d'] = df.groupby('Product_Category')['Total_Purchases'].transform(
            lambda x: x.rolling(7, min_periods=1).mean().fillna(method='ffill')
        )
        df['Rolling_14d'] = df.groupby('Product_Category')['Total_Purchases'].transform(
            lambda x: x.rolling(14, min_periods=1).mean().fillna(method='ffill')
        )
        df['Segment_Product'] = df['Customer_Segment'] * df['Product_Category']

        Config.PURCHASE_THRESHOLD = df['Total_Purchases'].median()

        self._preprocessed_data = df
        self._X = df[Config.FEATURE_COLUMNS]
        self._y = (df['Total_Purchases'] > Config.PURCHASE_THRESHOLD).astype(int)
        return self._X, self._y

    def get_cluster_data(self):
        df = self._preprocessed_data.copy()
        cluster_df = df[Config.CLUSTER_FEATURES]
        cluster_df['Income'] = cluster_df['Income'].astype('category').cat.codes
        self.scaler = StandardScaler()
        cluster_df[cluster_df.select_dtypes(['float','int']).columns] = self.scaler.fit_transform(
            cluster_df.select_dtypes(['float','int']).fillna(0)
        )
        return cluster_df

    def save_preprocessor(self):
        self._preprocessed_data = None
        self._X = None
        self._y = None
        joblib.dump(self, Config.PREPROCESSOR_PATH, compress=3)

class ModelTrainer:
    def __init__(self, preprocessor: DataPreprocessor):
        self.preprocessor = preprocessor
        self.model = None
        self.cluster_model = None
        self.cluster_labels = None

    def train_predictive_model(self):
        X, y = self.preprocessor.load_and_preprocess()
        X = X.loc[:, X.nunique() > 1]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
        )
        base_model = lgb.LGBMClassifier(random_state=Config.RANDOM_STATE, n_estimators=300, learning_rate=0.05,
                                        max_depth=8, num_leaves=64, is_unbalance=True)
        base_model.fit(X_train, y_train)

        self.model = base_model
        joblib.dump(self.model, Config.MODEL_PATH)

        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        acc = accuracy_score(y_test, y_pred) * 100

        print("\nMetrics for : Sales Forecast Dashboard")
        print("=== Press Detected for Forecast ===\n")
        print(f"MAE          : {mae:.2f}")
        print(f"RMSE         : {rmse:.2f}")
        print(f"Accuracy (%) : {acc:.2f}")
        print("\n===============================\n")

        from sklearn.metrics import precision_score, recall_score, f1_score
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"✅ Model trained successfully! | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}%")

    def train_clustering_model(self):
        cluster_data = self.preprocessor.get_cluster_data()
        self.cluster_model = KMeans(n_clusters=Config.N_CLUSTERS, random_state=Config.RANDOM_STATE)
        self.cluster_model.fit(cluster_data)
        self.cluster_labels = self.cluster_model.labels_
        joblib.dump(self.cluster_model, Config.CLUSTER_MODEL_PATH)

    def save_preprocessor(self):
        self.preprocessor.save_preprocessor()

class InsightsGenerator:
    def __init__(self, preprocessor: DataPreprocessor, trainer: ModelTrainer):
        self.preprocessor = preprocessor
        self.trainer = trainer

    def generate_descriptive_insights(self) -> str:
        df = self.preprocessor._preprocessed_data
        insights = []
        insights.append("### Demographic Insights")
        insights.append(f"- Average Age: {df['Age'].mean():.2f}")
        insights.append(f"- Gender Distribution: {df['Gender'].value_counts(normalize=True).to_dict()}")
        insights.append(f"- Income Levels: {df['Income'].value_counts(normalize=True).to_dict()}")
        insights.append("\n### Purchase Behavior")
        insights.append(f"- Average Total Purchases: {df['Total_Purchases'].mean():.2f}")
        insights.append(f"- Average Amount: {df['Amount'].mean():.2f}")
        insights.append(f"- Top Product Categories: {df['Product_Category'].value_counts().head(3).to_dict()}")
        insights.append("\n### Temporal Patterns")
        insights.append(f"- Peak Hour: {df['Hour'].mode()[0]}")
        insights.append(f"- Peak Month: {df['Month'].mode()[0]}")
        return "\n".join(insights)

    def generate_segmentation_insights(self) -> str:
        if self.trainer.cluster_labels is None:
            return "Clustering not performed yet."
        df = self.preprocessor._preprocessed_data.copy()
        df['Cluster'] = self.trainer.cluster_labels
        insights = []
        for cluster in range(Config.N_CLUSTERS):
            cluster_df = df[df['Cluster'] == cluster]
            insights.append(f"\n#### Segment {cluster + 1} ({len(cluster_df)} customers)")
            insights.append(f"- Avg Age: {cluster_df['Age'].mean():.2f}")
            insights.append(f"- Avg Purchases: {cluster_df['Total_Purchases'].mean():.2f}")
            insights.append(f"- Avg Ratings: {cluster_df['Ratings'].mean():.2f}")
            insights.append(f"- Common Income: {cluster_df['Income'].mode()[0]}")
        return "\n".join(insights)

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer(preprocessor)
    trainer.train_predictive_model()
    trainer.train_clustering_model()
    trainer.save_preprocessor()
    print("✓ Models, preprocessor, and clustering saved successfully!")
