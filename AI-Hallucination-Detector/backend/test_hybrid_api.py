import requests
import json
import time

url = "http://localhost:5001/detect"

print("--- TEST 1: FAST DEMO MODE (Green State) ---")
payload1 = {
    "query": "What is the recommended treatment for SARS-CoV-2?",
    "use_real_llm": False
}
try:
    start = time.time()
    res1 = requests.post(url, json=payload1)
    print(f"Time taken: {time.time() - start:.2f} seconds")
    data1 = res1.json()
    print("Graph Confidence:", data1.get('graph_confidence'))
    for r in data1.get('results', [])[:2]:  # Print first 2
        print(f"Status: {r['status']} | Confidence: {r['confidence']}% | Text: {r['text']}")
except Exception as e:
    print("Test 1 failed:", e)

print("\n--- TEST 2: LOCAL LLM INFERENCE (Real Generation) ---")
payload2 = {
    "query": "What are the common side effects of lisinopril?",
    "use_real_llm": True
}
try:
    start = time.time()
    res2 = requests.post(url, json=payload2)
    print(f"Time taken: {time.time() - start:.2f} seconds")
    data2 = res2.json()
    print("Graph Confidence:", data2.get('graph_confidence'))
    for r in data2.get('results', []):
        print(f"Status: {r['status']} | Confidence: {r['confidence']}% | Text: {r['text']}")
except Exception as e:
    print("Test 2 failed:", e)
