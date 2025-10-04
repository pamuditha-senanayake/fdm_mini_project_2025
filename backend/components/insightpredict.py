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
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path
import io
from base64 import b64encode
import numpy as np
import joblib
import os


# -----------------------------
# Configuration
# -----------------------------
class Config:
    """Configuration class for the application."""
    DATA_PATH: str = "../data.csv"
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42 
    MODEL_PATH: str = "predictive_model.joblib"
    CLUSTER_MODEL_PATH: str = "clustering_model.joblib"
    PREPROCESSOR_PATH: str = "preprocessor.joblib"
    MODEL_PARAMS: dict = {
        'n_estimators': [500, 1000],
        'learning_rate': [0.01, 0.05],
        'max_depth': [5, 10],
        'num_leaves': [31, 64]
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
    """Handles data loading, preprocessing, and feature engineering."""
    
    def __init__(self):
        self.category_mappings: Dict[str, Dict] = {}
        self.config = Config()
        self.data_path = self._resolve_data_path()
        self.scaler = StandardScaler()
        self._preprocessed_data: pd.DataFrame = None
        self._X: pd.DataFrame = None
        self._y: pd.Series = None

    def _resolve_data_path(self) -> Path:
        """Resolve the path to data.csv robustly regardless of CWD.

        Search order:
        1) DATA_CSV environment variable (absolute or relative to this file)
        2) Config.DATA_PATH relative to this file
        3) ./data.csv next to this file
        4) ../data.csv (one directory up)
        5) ../../data.csv (two directories up)
        """
        # If env var is set, prefer it
        env_path = os.getenv("DATA_CSV")
        base_dir = Path(__file__).resolve().parent

        candidate_paths = []

        if env_path:
            env_p = Path(env_path)
            if not env_p.is_absolute():
                env_p = base_dir / env_p
            candidate_paths.append(env_p)

        # Config path relative to this file
        cfg_path = Path(self.config.DATA_PATH)
        if not cfg_path.is_absolute():
            cfg_path = base_dir / cfg_path
        candidate_paths.append(cfg_path)

        # Common fallbacks
        candidate_paths.extend([
            base_dir / "data.csv",
            base_dir.parent / "data.csv",
            base_dir.parent.parent / "data.csv",
        ])

        for p in candidate_paths:
            if p.exists():
                return p

        # If nothing found, fall back to original config path (even if missing) for clearer error
        return candidate_paths[0]

    def load_and_preprocess(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and preprocess the dataset with NaN handling."""
        if self._preprocessed_data is not None:
            return self._X, self._y
        
        print("Loading data...")
        # Resolve again in case env or files changed between runs
        self.data_path = self._resolve_data_path()
        if not self.data_path.exists():
            raise FileNotFoundError(f"Could not locate data.csv. Tried: {self.data_path}")
        print(f"Reading CSV from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Handle missing values
        print("Handling missing values...")
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].mean())
            elif col in self.config.CATEGORICAL_COLUMNS:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # Replace invalid time values
        df['Time'] = df['Time'].replace('######', np.nan)
        df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.hour
        df['Hour'] = df['Hour'].fillna(df['Hour'].mean())

        # Process categorical variables
        print("Encoding categorical variables...")
        for col in self.config.CATEGORICAL_COLUMNS:
            df[col] = df[col].astype('category')
            self.category_mappings[col] = dict(enumerate(df[col].cat.categories))
            self.category_mappings[f"{col}_inv"] = {v: k for k, v in self.category_mappings[col].items()}
            df[col] = df[col].cat.codes

        # Process datetime features
        print("Extracting datetime features...")
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Day'] = df['Date'].dt.day
        df['Weekday'] = df['Date'].dt.weekday
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month

        # Sort and compute rolling averages
        print("Computing rolling features...")
        df = df.sort_values('Date')
        df['Rolling_7d'] = df.groupby('Product_Category')['Total_Purchases'].transform(
            lambda x: x.rolling(7, min_periods=1).mean().fillna(method='ffill')
        )
        df['Rolling_14d'] = df.groupby('Product_Category')['Total_Purchases'].transform(
            lambda x: x.rolling(14, min_periods=1).mean().fillna(method='ffill')
        )
        df['Segment_Product'] = df['Customer_Segment'] * df['Product_Category']

        # Set purchase threshold as median
        self.config.PURCHASE_THRESHOLD = df['Total_Purchases'].median()
        print(f"Purchase threshold set to: {self.config.PURCHASE_THRESHOLD}")
        
        self._preprocessed_data = df
        self._X = df[self.config.FEATURE_COLUMNS]
        self._y = (df['Total_Purchases'] > self.config.PURCHASE_THRESHOLD).astype(int)
        
        print(f"Preprocessing complete. Dataset shape: {df.shape}")
        return self._X, self._y

    def get_cluster_data(self) -> pd.DataFrame:
        """Prepare data for clustering with NaN handling."""
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
    """Handles model training for both prediction and clustering."""
    
    def __init__(self, preprocessor: DataPreprocessor):
        self.preprocessor = preprocessor
        self.model = None
        self.metrics = {}
        self.cluster_model = None
        self.cluster_labels = None

    def train_predictive_model(self) -> None:
        """Train the classification model with hyperparameter tuning."""
        print("\n" + "="*50)
        print("Training Predictive Model...")
        print("="*50)
        
        X, y = self.preprocessor.load_and_preprocess()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config().TEST_SIZE, random_state=Config().RANDOM_STATE
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        base_model = lgb.LGBMClassifier(random_state=Config().RANDOM_STATE)
        grid_search = GridSearchCV(
            base_model, Config().MODEL_PARAMS, cv=5, scoring='accuracy', n_jobs=-1, verbose=0
        )
        
        print("\nPerforming grid search...")
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        
        y_pred = self.model.predict(X_test)
        self.metrics['accuracy'] = accuracy_score(y_test, y_pred)
        self.metrics['r2'] = r2_score(y_test, y_pred)
        self.metrics['mae'] = mean_absolute_error(y_test, y_pred)

        print(f"\nModel Performance:")
        print(f"  Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"  R²: {self.metrics['r2']:.4f}")
        print(f"  MAE: {self.metrics['mae']:.4f}")

        # Save the trained predictive model
        joblib.dump(self.model, Config().MODEL_PATH)
        print(f"\n✓ Model saved to '{Config().MODEL_PATH}'")

    def train_clustering_model(self) -> None:
        """Train K-Means clustering for customer segmentation."""
        print("\n" + "="*50)
        print("Training Clustering Model...")
        print("="*50)
        
        cluster_data = self.preprocessor.get_cluster_data()
        print(f"Clustering {len(cluster_data)} customers into {Config().N_CLUSTERS} segments...")
        
        self.cluster_model = KMeans(n_clusters=Config().N_CLUSTERS, random_state=Config().RANDOM_STATE)
        self.cluster_labels = self.cluster_model.fit_predict(cluster_data)

        # Save the trained clustering model
        joblib.dump(self.cluster_model, Config().CLUSTER_MODEL_PATH)
        print(f"✓ Clustering model saved to '{Config().CLUSTER_MODEL_PATH}'")

    def save_preprocessor(self) -> None:
        """Save the preprocessor for later use."""
        joblib.dump(self.preprocessor, Config().PREPROCESSOR_PATH)
        print(f"✓ Preprocessor saved to '{Config().PREPROCESSOR_PATH}'")


# -----------------------------
# Insights Generator
# -----------------------------
class InsightsGenerator:
    """Generates insights and visualizations from the data."""
    
    def __init__(self, preprocessor: DataPreprocessor, trainer: ModelTrainer):
        self.preprocessor = preprocessor
        self.trainer = trainer

    def generate_descriptive_insights(self) -> str:
        """Generate descriptive statistics and insights."""
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
        """Generate insights from clustering."""
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
# Predictor
# -----------------------------
class Predictor:
    """Handles predictions for new data."""
    
    def __init__(self, trainer: ModelTrainer):
        self.trainer = trainer
        self.preprocessor = trainer.preprocessor
        self.config = Config()
        self._cached_means = None

    def _get_cached_means(self) -> Dict[str, float]:
        """Cache mean values for performance."""
        if self._cached_means is None:
            X, _ = self.preprocessor.load_and_preprocess()
            self._cached_means = {
                'Total_Amount': X['Total_Amount'].mean(),
                'Rolling_7d': X['Rolling_7d'].mean(),
                'Rolling_14d': X['Rolling_14d'].mean()
            }
        return self._cached_means


# -----------------------------
# Model Loader
# -----------------------------
def load_models():
    """Load saved models and preprocessor."""
    if not os.path.exists(Config().MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {Config().MODEL_PATH}")
    if not os.path.exists(Config().CLUSTER_MODEL_PATH):
        raise FileNotFoundError(f"Clustering model file not found: {Config().CLUSTER_MODEL_PATH}")
    if not os.path.exists(Config().PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessor file not found: {Config().PREPROCESSOR_PATH}")
    
    print("\nLoading saved models...")
    preprocessor = joblib.load(Config().PREPROCESSOR_PATH)
    model = joblib.load(Config().MODEL_PATH)
    cluster_model = joblib.load(Config().CLUSTER_MODEL_PATH)
    
    print("✓ Models loaded successfully!")
    
    return preprocessor, model, cluster_model


# -----------------------------
# Gradio UI
# -----------------------------
def create_ui() -> gr.Blocks:
    """Create Gradio interface with Get Insights button."""
    
    def get_insights():
        """Generate and return insights when button is clicked."""
        try:
            # Load models
            preprocessor, model, cluster_model = load_models()
            
            # Create trainer object with loaded models
            trainer = ModelTrainer(preprocessor)
            trainer.model = model
            trainer.cluster_model = cluster_model
            
            # Load preprocessed data and cluster labels
            X, y = preprocessor.load_and_preprocess()
            cluster_data = preprocessor.get_cluster_data()
            trainer.cluster_labels = cluster_model.predict(cluster_data)
            
            # Calculate metrics
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=Config().TEST_SIZE, random_state=Config().RANDOM_STATE
            )
            y_pred = model.predict(X_test)
            trainer.metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred)
            }
            
            # Generate insights
            insights_gen = InsightsGenerator(preprocessor, trainer)
            
            metrics_text = (f"**Model Performance:** "
                          f"Accuracy = {round(trainer.metrics['accuracy'] * 100, 2)}%, "
                          f"R² = {round(trainer.metrics['r2'], 3)}, "
                          f"MAE = {round(trainer.metrics['mae'], 3)}")
            
            desc_insights = insights_gen.generate_descriptive_insights()
            seg_insights = insights_gen.generate_segmentation_insights()
            
            return metrics_text, desc_insights, seg_insights
            
        except FileNotFoundError as e:
            error_msg = f"❌ Error: {str(e)}\n\nPlease train the models first by running the training script."
            return error_msg, "", ""
        except Exception as e:
            error_msg = f"❌ Error generating insights: {str(e)}"
            return error_msg, "", ""
    
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
    """Train and save all models."""
    print("\n" + "="*50)
    print("SALES PREDICTION & CUSTOMER INSIGHTS APP")
    print("TRAINING MODE")
    print("="*50 + "\n")
    
    # Initialize components
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer(preprocessor)
    
    # Train models
    trainer.train_predictive_model()
    trainer.train_clustering_model()
    
    # Save preprocessor
    trainer.save_preprocessor()
    
    print("\n" + "="*50)
    print("✓ All models trained and saved successfully!")
    print("="*50 + "\n")


def launch_app():
    """Launch the Gradio interface."""
    print("\n" + "="*50)
    print("Launching Gradio Interface...")
    print("="*50 + "\n")
    
    demo = create_ui()
    demo.launch()


if __name__ == "__main__":
    # First, train the models
    train_models()
    
    # Then launch the app
    launch_app()