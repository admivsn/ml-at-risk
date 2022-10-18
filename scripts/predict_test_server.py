import json
import requests

predict_json = {'instances': [[50, 20], [700, 300]]}

# Mock input data as list of dicts
inputs = [
    {
        'deposits': 50,
        'withdrawals': 0,
        'stakes': 20,
        'product': 'bet',
        'age': '50+',
    },
    {
        'deposits': 700,
        'withdrawals': 0,
        'stakes': 300,
        'product': 'bet',
        'age': '<30',
    },
]

response = requests.post(
    url="http://localhost:8080/predict",
    data=json.dumps(inputs),
    headers={'content-type': 'application/json', 'charset': 'utf-8'}
)
print(response.json())
