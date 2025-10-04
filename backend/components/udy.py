import gradio as gr
import joblib
import os
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from train_models import DataPreprocessor, ModelTrainer, Config, InsightsGenerator

def load_models():
    for path in [Config.MODEL_PATH, Config.CLUSTER_MODEL_PATH, Config.PREPROCESSOR_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    preprocessor = joblib.load(Config.PREPROCESSOR_PATH)
    preprocessor.load_and_preprocess()
    model = joblib.load(Config.MODEL_PATH)
    cluster_model = joblib.load(Config.CLUSTER_MODEL_PATH)
    return preprocessor, model, cluster_model

def get_insights():
    try:
        preprocessor, model, cluster_model = load_models()
        trainer = ModelTrainer(preprocessor)
        trainer.model = model
        cluster_data = preprocessor.get_cluster_data()
        trainer.cluster_model = cluster_model
        trainer.cluster_labels = cluster_model.predict(cluster_data)

        X, y = preprocessor.load_and_preprocess()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
        )
        y_pred = model.predict(X_test)
        trainer.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred)
        }

        insights_gen = InsightsGenerator(preprocessor, trainer)
        metrics_text = (f"**Model Performance:** Accuracy = {trainer.metrics['accuracy']*100:.2f}%, "
                        f"R² = {trainer.metrics['r2']:.3f}, MAE = {trainer.metrics['mae']:.3f}")
        desc_insights = insights_gen.generate_descriptive_insights()
        seg_insights = insights_gen.generate_segmentation_insights()
        return metrics_text, desc_insights, seg_insights

    except Exception as e:
        return f"❌ Error: {e}", "", ""

# Gradio UI
def launch_app():
    with gr.Blocks(title="Sales Insights App") as demo:
        gr.Markdown("## 🛒 Sales Prediction & Customer Insights App")
        with gr.Row():
            btn = gr.Button("🔍 Get Insights")
        metrics_out = gr.Markdown("")
        desc_out = gr.Markdown("")
        seg_out = gr.Markdown("")
        btn.click(fn=get_insights, outputs=[metrics_out, desc_out, seg_out])
    demo.launch(share=True, prevent_thread_lock=True)

if __name__ == "__main__":
    launch_app()
