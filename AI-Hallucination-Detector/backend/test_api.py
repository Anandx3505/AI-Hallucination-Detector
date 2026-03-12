import requests
import json

url = "http://localhost:5001/detect"
payload = {"query": "What is the recommended treatment for severe acute respiratory syndrome?"}

print(f"Sending request to {url}...")
try:
    response = requests.post(url, json=payload, timeout=10)
    print("Status Code:", response.status_code)
    if response.status_code == 200:
        print("Response:", json.dumps(response.json(), indent=2))
    else:
        print("Error Response:", response.text)
except requests.exceptions.RequestException as e:
    print("Connection Error:", e)
