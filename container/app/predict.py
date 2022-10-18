import joblib
import uvicorn
import pandas as pd
from fastapi import FastAPI, Request


# Initialise app
app = FastAPI()


# Initialise an empty dict to store model and override on startup
model = {}
@app.on_event('startup')
def load_model():
    """
    Add method to load model on startup.
    """
    model["clf"] = joblib.load("/app/data/model.joblib")


# Add post predict method
@app.post("/predict")
async def predict(request: Request):
    """
    Create an endpoint for generating predictions.
    """
    # wait for data passed to the endpoint
    request_json = await request.json()
    instances = request_json["instances"]

    col_names = ['deposits', 'stakes']
    inputs = pd.DataFrame(instances, columns=col_names)

    outputs = model["clf"].predict(inputs)
    return {"predictions": outputs.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, debug=True)
