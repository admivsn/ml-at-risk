import joblib
import uvicorn
import pandas as pd
from fastapi import FastAPI, Request

# app -----------------------

app = FastAPI()


@app.post("/predict")
async def predict(request: Request):
    """
    Create an endpoint for generating predictions
    """

    model = joblib.load("/app/data/model.joblib")

    # wait for data passed to the endpoint
    request_json = await request.json()
    instances = request_json["instances"]

    col_names = ['deposits', 'stakes']
    inputs = pd.DataFrame(instances, columns=col_names)

    outputs = model.predict(inputs)
    return {"predictions": outputs.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, debug=True)
