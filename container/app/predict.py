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
    input = await request.json()
    
    # Convert list of dicts to dataframe
    df_input = pd.DataFrame(input)

    # Predict each input row
    df_predictions = model["clf"].predict(df_input)

    # Output as a list
    return df_predictions.tolist()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, debug=True)
