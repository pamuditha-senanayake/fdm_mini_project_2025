"""
Sales Prediction & Customer Insights Application
A complete ML application for sales analysis, customer segmentation, and insights generation.
"""

import pandas as pd
import lightgbm as lgb
import gradio as gr
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os
from pathlib import Path
from typing import Dict, List, Tuple

# -----------------------------
# Configuration
# -----------------------------
class Config:
    DATA_PATH: str = "data.csv"
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    MODEL_PATH: str = "../predictive_model.joblib"
    CLUSTER_MODEL_PATH: str = "../clustering_model.joblib"
    PREPROCESSOR_PATH: str = "../preprocessor.joblib"
    MODEL_PARAMS: dict = {
        'n_estimators': [200, 500],
        'learning_rate': [0.01, 0.05],
        'max_depth': [3, 5, 7],
        'num_leaves': [15, 31, 63]
    }
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
    N_CLUSTERS: int = 4
    PURCHASE_THRESHOLD: float = None


# -----------------------------
# Data Preprocessor
# -----------------------------
class DataPreprocessor:
    def __init__(self):
        self.category_mappings: Dict[str, Dict] = {}
        self.config = Config()
        self.data_path = self._resolve_data_path()
        self.scaler = StandardScaler()
        self._preprocessed_data: pd.DataFrame = None
        self._X: pd.DataFrame = None
        self._y: pd.Series = None

    def _resolve_data_path(self) -> Path:
        base_dir = Path("..").resolve()
        candidate_paths = [
            Path(self.config.DATA_PATH),
            base_dir / "data.csv",
            base_dir.parent / "data.csv"
        ]
        for p in candidate_paths:
            if p.exists():
                return p
        return candidate_paths[0]

    def load_and_preprocess(self) -> Tuple[pd.DataFrame, pd.Series]:
        if self._preprocessed_data is not None:
            return self._X, self._y

        df = pd.read_csv(self.data_path)

        # Fill missing values
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].mean())
            elif col in self.config.CATEGORICAL_COLUMNS:
                df[col] = df[col].fillna(df[col].mode()[0])

        # Time features
        df['Time'] = df.get('Time', '00:00:00').replace('######', np.nan)
        df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.hour
        df['Hour'] = df['Hour'].fillna(df['Hour'].mean())

        # Encode categorical
        for col in self.config.CATEGORICAL_COLUMNS:
            df[col] = df[col].astype('category')
            self.category_mappings[col] = dict(enumerate(df[col].cat.categories))
            self.category_mappings[f"{col}_inv"] = {v: k for k, v in self.category_mappings[col].items()}
            df[col] = df[col].cat.codes

        # Date features
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Day'] = df['Date'].dt.day
        df['Weekday'] = df['Date'].dt.weekday
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month

        # Rolling features
        df = df.sort_values('Date')
        df['Rolling_7d'] = df.groupby('Product_Category')['Total_Purchases'].transform(
            lambda x: x.rolling(7, min_periods=1).mean().fillna(method='ffill')
        )
        df['Rolling_14d'] = df.groupby('Product_Category')['Total_Purchases'].transform(
            lambda x: x.rolling(14, min_periods=1).mean().fillna(method='ffill')
        )
        df['Segment_Product'] = df['Customer_Segment'] * df['Product_Category']

        self.config.PURCHASE_THRESHOLD = df['Total_Purchases'].median()

        self._preprocessed_data = df
        self._X = df[self.config.FEATURE_COLUMNS]
        self._y = (df['Total_Purchases'] > self.config.PURCHASE_THRESHOLD).astype(int)

        return self._X, self._y

    def get_cluster_data(self) -> pd.DataFrame:
        df = self._preprocessed_data.copy()
        cluster_df = df[self.config.CLUSTER_FEATURES]
        cluster_df['Income'] = cluster_df['Income'].astype('category').cat.codes
        numerical_cols = cluster_df.select_dtypes(include=['float', 'int']).columns
        cluster_df[numerical_cols] = self.scaler.fit_transform(cluster_df[numerical_cols].fillna(0))
        return cluster_df


# -----------------------------
# Model Trainer
# -----------------------------
class ModelTrainer:
    def __init__(self, preprocessor: DataPreprocessor):
        self.preprocessor = preprocessor
        self.model = None
        self.metrics = {}
        self.cluster_model = None
        self.cluster_labels = None

    def train_predictive_model(self) -> None:
        X, y = self.preprocessor.load_and_preprocess()
        X = X.loc[:, X.nunique() > 1]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config().TEST_SIZE, random_state=Config().RANDOM_STATE
        )
        base_model = lgb.LGBMClassifier(random_state=Config().RANDOM_STATE, verbose=-1)
        grid_search = GridSearchCV(
            base_model, Config().MODEL_PARAMS, cv=3, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        y_pred = self.model.predict(X_test)
        self.metrics['accuracy'] = accuracy_score(y_test, y_pred)
        self.metrics['r2'] = r2_score(y_test, y_pred)
        self.metrics['mae'] = mean_absolute_error(y_test, y_pred)
        joblib.dump(self.model, Config().MODEL_PATH)

    def train_clustering_model(self) -> None:
        cluster_data = self.preprocessor.get_cluster_data()
        self.cluster_model = KMeans(n_clusters=Config().N_CLUSTERS, random_state=Config().RANDOM_STATE)
        self.cluster_labels = self.cluster_model.fit_predict(cluster_data)
        joblib.dump(self.cluster_model, Config().CLUSTER_MODEL_PATH)

    def save_preprocessor(self) -> None:
        joblib.dump(self.preprocessor, Config().PREPROCESSOR_PATH)


# -----------------------------
# Insights Generator
# -----------------------------
class InsightsGenerator:
    def __init__(self, preprocessor: DataPreprocessor, trainer: ModelTrainer):
        self.preprocessor = preprocessor
        self.trainer = trainer

    def generate_descriptive_insights(self) -> str:
        df = self.preprocessor._preprocessed_data
        insights = []
        insights.append("### Demographics")
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
        insights.append("### Customer Segments")
        for cluster in range(Config().N_CLUSTERS):
            cluster_df = df[df['Cluster'] == cluster]
            insights.append(f"\n#### Segment {cluster + 1} ({len(cluster_df)} customers)")
            insights.append(f"- Avg Age: {cluster_df['Age'].mean():.2f}")
            insights.append(f"- Avg Purchases: {cluster_df['Total_Purchases'].mean():.2f}")
            insights.append(f"- Avg Ratings: {cluster_df['Ratings'].mean():.2f}")
            insights.append(f"- Common Income: {cluster_df['Income'].mode()[0]}")
        return "\n".join(insights)


# -----------------------------
# Model Loader
# -----------------------------
def load_models():
    for path in [Config().MODEL_PATH, Config().CLUSTER_MODEL_PATH, Config().PREPROCESSOR_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    preprocessor = joblib.load(Config().PREPROCESSOR_PATH)
    preprocessor.load_and_preprocess()
    model = joblib.load(Config().MODEL_PATH)
    cluster_model = joblib.load(Config().CLUSTER_MODEL_PATH)
    return preprocessor, model, cluster_model


# -----------------------------
# Gradio UI
# -----------------------------
def create_ui() -> gr.Blocks:
    def get_insights():
        try:
            preprocessor, model, cluster_model = load_models()
            trainer = ModelTrainer(preprocessor)
            trainer.model = model
            cluster_data = preprocessor.get_cluster_data()
            trainer.cluster_labels = cluster_model.predict(cluster_data)
            trainer.cluster_model = cluster_model

            X, y = preprocessor.load_and_preprocess()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=Config().TEST_SIZE, random_state=Config().RANDOM_STATE
            )
            y_pred = model.predict(X_test)
            trainer.metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred)
            }

            insights_gen = InsightsGenerator(preprocessor, trainer)
            metrics_text = (f"**Model Performance:** "
                            f"Accuracy = {round(trainer.metrics['accuracy']*100,2)}%, "
                            f"R² = {round(trainer.metrics['r2'],3)}, "
                            f"MAE = {round(trainer.metrics['mae'],3)}")
            desc_insights = insights_gen.generate_descriptive_insights()
            seg_insights = insights_gen.generate_segmentation_insights()
            return metrics_text, desc_insights, seg_insights

        except FileNotFoundError as e:
            return f"❌ Error: {e}", "", ""
        except Exception as e:
            return f"❌ Unexpected error: {e}", "", ""

    with gr.Blocks(title="Sales Insights App") as demo:
        gr.Markdown("## 🛒 Sales Prediction & Customer Insights App")
        with gr.Row():
            get_insights_btn = gr.Button("🔍 Get Insights", variant="primary", size="lg")
        metrics_output = gr.Markdown("")
        desc_insights_output = gr.Markdown("")
        seg_insights_output = gr.Markdown("")
        get_insights_btn.click(
            fn=get_insights,
            outputs=[metrics_output, desc_insights_output, seg_insights_output]
        )
    return demo


# -----------------------------
# Main Execution
# -----------------------------
def train_models():
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer(preprocessor)
    trainer.train_predictive_model()
    trainer.train_clustering_model()
    trainer.save_preprocessor()
    print("✓ All models trained and saved successfully!")


def launch_app():
    demo = create_ui()
    demo.launch(share=True, prevent_thread_lock=True)


if __name__ == "__main__":
    train_models()
    launch_app()
