import pickle
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn 

app =FastAPI

@app.get('/'):
def my_function(text: str):
    return JSONResponse({"prediction": text})

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 9000)



model_path = "models/xgb_predictor.pkl"
with open(model_path, 'rb') as file:
    classifier = pickle.load(file)

