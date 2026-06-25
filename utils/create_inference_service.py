import requests
from azure.identity import AzureCliCredential

credentials = AzureCliCredential(tenant_id="<your tenant id>")
access_token = credentials.get_token("https://management.azure.com/.default")

headers = {
    "Authorization": f"Bearer {access_token.token}",
    "Content-Type": "application/json"
}

# NOTE: Update these variables with your account details
subscription_id = "" # TODO: replace with your subscription ID
resource_group = ""  # TODO: replace with your resource group name
region = ""  # TODO: replace with your region. We currently support regions: "eastus2", "westus3"
account_name = "" # TODO: replace with your account name

# 1. Request to create an account - only needs to be done once per account
request_payload = {
    "location": region
}

base_url = "https://management.azure.com"
response = requests.put(
    url = f"{base_url}/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.InferenceService/inferenceAccounts/{account_name}?api-version=2026-01-15-preview",
    headers=headers,
    json=request_payload
)

if response.status_code != 200:
    print(f"Failed to register account. Status code: {response.status_code}, Response: {response.text}")
else:
    print(f"Account registration successful.")
    print(f"Response: {response.json()}")