import json
import requests

predict_json = {'instances': [[50, 20], [700, 300]]}

# Mock input data as list of dicts
inputs = [
    {
        'deposits': 50,
        'stakes': 20,
    },
    {
        'deposits': 700,
        'stakes': 300,
    },
]

response = requests.post(
    url="http://localhost:8080/predict",
    data=json.dumps(inputs),
    headers={'content-type': 'application/json', 'charset': 'utf-8'}
)
print(response.json())
