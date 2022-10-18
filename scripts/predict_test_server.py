import json
import requests

predict_json = {'instances': [[50, 20], [700, 300]]}

headers = {'content-type': 'application/json', 'charset': 'utf-8'}

response = requests.post("http://localhost:8080/predict",
                         data=json.dumps(predict_json),
                         headers=headers)

print(response.json())
