import requests
import time

from azure.identity import AzureCliCredential

def get_access_token():
    credentials = AzureCliCredential(tenant_id="<your tenant id>")
    access_token = credentials.get_token("https://dbinference.azure.com")
    return access_token

headers = {
    "Authorization": f"Bearer {get_access_token().token}",
    "Content-Type": "application/json"
}

body = {
    "query": "What is the capital of France?",
    "documents": [
        "Berlin is the capital of Germany.",
        "Paris is the capital of France.",
        "Madrid is the capital of Spain."
    ],
    "return_documents": False,
    "top_k": 3,
    "batch_size": 32,
}

response = requests.post(
    "https://<your account name>.<your region>.dbinference.azure.com:443/inference/semanticReranking", 
    headers=headers, 
    json=body)

if response.status_code != 200:
    print(f"Failed to get inference. Status code: {response.status_code}, Response: {response.text}")
else:
    print(f"Inference successful. Response: {response.json()}")