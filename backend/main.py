# backend/main.py
import os
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error

from components.train_models import DataPreprocessor, ModelTrainer, Config, InsightsGenerator
from components import stockpredict, segmentpredict
from components import form_component
from components import promotionpredict
from fastapi import Body
import numpy as np
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
Config.MODEL_PATH = os.path.join(BASE_DIR, "components", "predictive_model.joblib")
Config.CLUSTER_MODEL_PATH = os.path.join(BASE_DIR, "components", "clustering_model.joblib")
Config.DATA_PATH = os.path.join(BASE_DIR, "components", "..", "data.csv")  # adjust if needed


def clean_json_floats(obj):
    if isinstance(obj, dict):
        return {k: clean_json_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_floats(i) for i in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


app = FastAPI(title="RetailIQ Insights API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(stockpredict.router)
app.include_router(promotionpredict.router)

app.include_router(form_component.router)

def get_or_train_models():

    if os.path.exists(Config.MODEL_PATH) and os.path.exists(Config.CLUSTER_MODEL_PATH):
        model = joblib.load(Config.MODEL_PATH)
        cluster_model = joblib.load(Config.CLUSTER_MODEL_PATH)
        return model, cluster_model


    preprocessor = DataPreprocessor()
    X, y = preprocessor.load_and_preprocess()
    trainer = ModelTrainer(preprocessor)
    trainer.train_predictive_model()
    trainer.train_clustering_model()


    joblib.dump(trainer.model, Config.MODEL_PATH)
    joblib.dump(trainer.cluster_model, Config.CLUSTER_MODEL_PATH)

    return trainer.model, trainer.cluster_model

@app.get("/insights")
def get_comprehensive_insights():
    try:
        preprocessor = DataPreprocessor()
        X, y = preprocessor.load_and_preprocess()


        model, cluster_model = get_or_train_models()


        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
        )
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred)
        }
        metrics_text = (f"**Model Performance:** Accuracy = {metrics['accuracy']*100:.2f}%, "
                        f"R² = {metrics['r2']:.3f}, MAE = {metrics['mae']:.3f}")


        cluster_data = preprocessor.get_cluster_data()
        cluster_labels = cluster_model.predict(cluster_data)


        trainer = ModelTrainer(preprocessor)
        trainer.model = model
        trainer.cluster_model = cluster_model
        trainer.cluster_labels = cluster_labels

        insights_gen = InsightsGenerator(preprocessor, trainer)
        descriptive_insights = insights_gen.generate_descriptive_insights()
        segmentation_insights = insights_gen.generate_segmentation_insights()

        return clean_json_floats({
            "metrics": metrics_text,
            "descriptive": descriptive_insights,
            "segmentation": segmentation_insights
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

@app.get("/")
def root():
    return {"message": "RetailIQ backend is running."}

# @app.on_event("startup")
# def load_models_on_startup():
#
#     segmentpredict.load_and_train_segment_model()

@app.post("/api/predict-segment")
async def predict_segment(data: dict = Body(...)):
    try:
        age = data.get("age")
        income = data.get("income")
        total_purchases = data.get("total_purchases")
        amount = data.get("amount")

        if None in [age, income, total_purchases, amount]:
            raise HTTPException(status_code=400, detail="Missing one or more required fields")

        result = segmentpredict.predict_segment(age, income, total_purchases, amount)


        return clean_json_floats(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
