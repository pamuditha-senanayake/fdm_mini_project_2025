# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# from components.train_models import DataPreprocessor, ModelTrainer, Config, InsightsGenerator
from components import stockpredict
from b2 import promotionpredict

# -----------------------------
# Fix paths for saved models (inside components/)
# -----------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Config.MODEL_PATH = os.path.join(BASE_DIR, "components", "predictive_model.joblib")
# Config.CLUSTER_MODEL_PATH = os.path.join(BASE_DIR, "components", "clustering_model.joblib")
# Config.DATA_PATH = os.path.join(BASE_DIR, "components", "..", "data.csv")  # adjust if needed

# -----------------------------
# FastAPI Setup
# -----------------------------
app = FastAPI(title="RetailIQ Insights API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(stockpredict.router)
app.include_router(promotionpredict.router)

# -----------------------------
# Load or Train Models
# -----------------------------
# def get_or_train_models():
#     """Load existing models or train new ones if missing."""
#     if os.path.exists(Config.MODEL_PATH) and os.path.exists(Config.CLUSTER_MODEL_PATH):
#         model = joblib.load(Config.MODEL_PATH)
#         cluster_model = joblib.load(Config.CLUSTER_MODEL_PATH)
#         return model, cluster_model
#
#     # Train models if missing
#     preprocessor = DataPreprocessor()
#     X, y = preprocessor.load_and_preprocess()
#     trainer = ModelTrainer(preprocessor)
#     trainer.train_predictive_model()
#     trainer.train_clustering_model()
#
#     # Save models
#     joblib.dump(trainer.model, Config.MODEL_PATH)
#     joblib.dump(trainer.cluster_model, Config.CLUSTER_MODEL_PATH)
#
#     return trainer.model, trainer.cluster_model

# -----------------------------
# API Endpoint
# -----------------------------
# @app.get("/insights")
# def get_comprehensive_insights():
#     try:
#         preprocessor = DataPreprocessor()
#         X, y = preprocessor.load_and_preprocess()
#
#         # Load or train models
#         model, cluster_model = get_or_train_models()
#
#         # Predictive metrics
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
#         )
#         y_pred = model.predict(X_test)
#         metrics = {
#             'accuracy': accuracy_score(y_test, y_pred),
#             'r2': r2_score(y_test, y_pred),
#             'mae': mean_absolute_error(y_test, y_pred)
#         }
#         metrics_text = (f"**Model Performance:** Accuracy = {metrics['accuracy']*100:.2f}%, "
#                         f"R² = {metrics['r2']:.3f}, MAE = {metrics['mae']:.3f}")
#
#         # Cluster predictions
#         cluster_data = preprocessor.get_cluster_data()
#         cluster_labels = cluster_model.predict(cluster_data)
#
#         # Generate insights
#         trainer = ModelTrainer(preprocessor)
#         trainer.model = model
#         trainer.cluster_model = cluster_model
#         trainer.cluster_labels = cluster_labels
#
#         insights_gen = InsightsGenerator(preprocessor, trainer)
#         descriptive_insights = insights_gen.generate_descriptive_insights()
#         segmentation_insights = insights_gen.generate_segmentation_insights()
#
#         return {
#             "metrics": metrics_text,
#             "descriptive": descriptive_insights,
#             "segmentation": segmentation_insights
#         }
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal error: {e}")

# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def root():
    return {"message": "RetailIQ backend is running."}

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
