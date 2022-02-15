import pandas as pd
from fastapi import FastAPI
from inference_onnx import ColaONNXPredictor

app = FastAPI(
    title="MLOps Basics App",
    description="Text classification of grammatical acceptability",
)

# load model
model_path = "./models/model.onnx"
predictor = ColaONNXPredictor(model_path)


@app.get("/")
async def home_page():
    return "<h2>Welcome to the MLOps Basics App</h2>"


@app.get("/predict")
async def get_prediction(text: str):
    # API gets a text and returns a prediction
    result = predictor.predict(text)
    
    # return label with largest score
    df = pd.DataFrame.from_dict([result[0], result[1]])
    prediction = df.loc[df['score'].idxmax()]["label"]
    return prediction
