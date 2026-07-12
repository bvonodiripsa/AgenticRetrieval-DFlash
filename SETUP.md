# Setup Guide: AgenticRetrieval-DFlash on a New Azure VM

This guide walks through setting up the full AgenticRetrieval-DFlash environment on a fresh Azure VM, including vLLM with DFlash speculative decoding, the FastAPI web app, and Cosmos DB connectivity.

## Prerequisites

- Azure subscription with H100 GPU quota
- Azure Cosmos DB account with the `food` database populated (58K products, 892K KG triples)
- Azure CLI access and Entra ID credentials

## Step 1: Create the Azure VM

```bash
az vm create \
  --resource-group <your-rg> \
  --name <your-vm-name> \
  --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest \
  --size Standard_NC80adis_H100_v5 \
  --os-disk-size-gb 512 \
  --admin-username azureuser \
  --generate-ssh-keys
```

**Minimum hardware**: 2x NVIDIA H100 GPUs (80GB+ each) for tensor-parallel vLLM serving of Qwen3.5-27B with DFlash. A single H100 won't fit the model + KV cache.

| Spec | Value |
|------|-------|
| VM size | Standard_NC80adis_H100_v5 |
| GPUs | 2x NVIDIA H100 NVL (96GB each) |
| OS | Ubuntu 22.04 LTS |
| Disk | 512 GB (model weights need ~120GB cache) |
| CUDA | 13.0 |
| NVIDIA driver | 580+ |

## Step 2: Install NVIDIA Drivers and CUDA

If the VM image doesn't come with NVIDIA drivers pre-installed:

```bash
sudo apt-get update && sudo apt-get install -y build-essential

# Install NVIDIA driver (580+) and CUDA 13.0
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit

# Verify
nvidia-smi
```

You should see both H100 GPUs listed with driver 580+ and CUDA 13.0.

## Step 3: Install Python 3.12

The project uses Python 3.12 (Ubuntu 22.04 ships with 3.10):

```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
```

## Step 4: Clone the Repositories

```bash
cd /home/azureuser

# Main repo (KG + DFlash). The "Original" backend now runs from the in-repo
# dynamic_retriever/utils modules — no separate AgenticRetrieval clone needed.
git clone https://github.com/bvonodiripsa/AgenticRetrieval-DFlash.git
```

## Step 5: Create Virtual Environment and Install Dependencies

```bash
cd /home/azureuser/AgenticRetrieval-DFlash

python3.12 -m venv .venv
source .venv/bin/activate

# Install vLLM (includes PyTorch with CUDA)
pip install vllm==0.23.0

# Install application dependencies
pip install \
  azure-cosmos==4.16.1 \
  azure-identity==1.25.2 \
  openai==2.21.0 \
  fastapi==0.136.3 \
  uvicorn==0.49.0 \
  httpx==0.28.1 \
  aiohttp==3.13.3 \
  sentence-transformers==5.6.0 \
  pydantic==2.12.5 \
  pyyaml \
  numpy
```

### Key package versions

| Package | Version | Purpose |
|---------|---------|---------|
| vllm | 0.23.0 | LLM inference server with DFlash speculative decoding |
| torch | 2.11.0 | Deep learning framework (installed by vLLM) |
| azure-cosmos | 4.16.1 | Cosmos DB SDK (includes semantic reranker support) |
| azure-identity | 1.25.2 | Azure RBAC authentication |
| openai | 2.21.0 | OpenAI-compatible API client (for vLLM and Azure OpenAI) |
| fastapi | 0.136.3 | Web framework |
| sentence-transformers | 5.6.0 | In-process embedding model |
| transformers | 5.12.1 | Hugging Face model loading (installed by vLLM) |

## Step 6: Download Model Weights

This downloads ~55GB of model files. Takes ~30 minutes on first run.

```bash
source /home/azureuser/AgenticRetrieval-DFlash/.venv/bin/activate

python -c "
from huggingface_hub import snapshot_download
print('Downloading Qwen3.5-27B (~52GB)...')
snapshot_download('Qwen/Qwen3.5-27B')
print('Downloading DFlash draft model (~1GB)...')
snapshot_download('z-lab/Qwen3.5-27B-DFlash')
print('Downloading embedding model (~1.2GB)...')
snapshot_download('Qwen/Qwen3-Embedding-0.6B')
print('Done.')
"
```

Models are cached in `~/.cache/huggingface/hub/`.

## Step 7: Configure Azure CLI and Cosmos DB Access

### Install Azure CLI

```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

### Log in to Azure

```bash
az login --tenant 43083d15-7273-40c1-b7db-39efd9ccc17a
```

### Verify Cosmos DB connectivity

```bash
source /home/azureuser/AgenticRetrieval-DFlash/.venv/bin/activate

python -c "
import asyncio
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import AzureCliCredential
async def test():
    cred = AzureCliCredential(tenant_id='43083d15-7273-40c1-b7db-39efd9ccc17a')
    client = CosmosClient('https://divdet.documents.azure.com:443/', credential=cred)
    db = client.get_database_client('food')
    ctr = db.get_container_client('food')
    async for doc in ctr.query_items('SELECT TOP 1 c.id FROM c'):
        print('Cosmos DB OK:', doc)
        break
    await client.close()
    await cred.close()
asyncio.run(test())
"
```

### Required RBAC roles

Ask your admin to assign these roles on the `divdet` Cosmos DB account:

| Role | Scope | Purpose |
|------|-------|---------|
| Cosmos DB Built-in Data Contributor | `Microsoft.DocumentDB/databaseAccounts/divdet` | Read/write data |
| Semantic Reranker User | `Microsoft.InferenceService/inferenceAccounts/divdet` | Semantic reranking |

Assignment command for the admin:

```bash
# Data access
az cosmosdb sql role assignment create \
  --account-name divdet \
  --resource-group ams-cosmos-db \
  --role-definition-name "Cosmos DB Built-in Data Contributor" \
  --principal-id "<your-user-object-id>" \
  --scope "/"

# Semantic reranker (note: InferenceService scope, not DocumentDB)
az role assignment create \
  --role "Semantic Reranker User" \
  --assignee-object-id "<your-user-object-id>" \
  --assignee-principal-type "User" \
  --scope "/subscriptions/b7d41fc8-d35d-41db-92ed-1f7f1d32d4d9/resourceGroups/ams-cosmos-db/providers/Microsoft.InferenceService/inferenceAccounts/divdet"
```

## Step 8: Update Config Files (if using a different Cosmos DB account)

The config points to a Cosmos DB account and database. Copy the template and
edit your settings:

```bash
cd /home/azureuser/AgenticRetrieval-DFlash

# Create your config from the template, then set cosmos.*, embedding.*, llm.*
cp config.yaml.example my.yaml
vi my.yaml
```

Key fields to update:

```yaml
cosmos:
  uri: "https://<your-account>.documents.azure.com:443/"
  database_name: "<your-database>"
  tenant_id: "<your-tenant-id>"
```

## Step 9: Start vLLM with DFlash

```bash
cd /home/azureuser/AgenticRetrieval-DFlash
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3.5-27B \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.92 \
  --dtype float16 \
  --quantization fp8 \
  --spec-model z-lab/Qwen3.5-27B-DFlash \
  --spec-tokens 5 \
  --enable-prefix-caching \
  --port 8000
```

**First startup takes 10-15 minutes** (model loading + torch.compile + CUDA graph capture). Subsequent starts use cached compilations (~3-5 min).

Wait until you see `Application startup complete`, then verify:

```bash
curl http://localhost:8000/v1/models
```

### vLLM argument reference

| Argument | Value | Purpose |
|----------|-------|---------|
| `--tensor-parallel-size 2` | 2 | Split model across 2 H100 GPUs |
| `--max-model-len 16384` | 16K | Maximum sequence length |
| `--gpu-memory-utilization 0.92` | 92% | GPU memory for KV cache |
| `--dtype float16` | FP16 | Computation precision |
| `--quantization fp8` | FP8 | Weight quantization (halves memory) |
| `--spec-model` | `z-lab/Qwen3.5-27B-DFlash` | DFlash draft model for speculative decoding |
| `--spec-tokens 5` | 5 | Draft tokens per speculative step |
| `--enable-prefix-caching` | — | Cache common prompt prefixes |

## Step 10: Start the Web Application

In a **separate terminal**:

```bash
cd /home/azureuser/AgenticRetrieval-DFlash
source .venv/bin/activate

# The Cosmos reranker endpoint comes from cosmos.semantic_reranker_endpoint in my.yaml.
python api.py --config my.yaml --host localhost --port 8080
```

Wait for `Application startup complete`, then verify:

```bash
curl http://localhost:8080/health
# Should return: {"status":"ok"}
```

## Step 11: Open the Firewall (NSG Rules)

Open port 8080 from your IP. You may need rules on **both** the NIC-level and subnet-level NSGs:

```bash
# NIC-level NSG
az network nsg rule create \
  --nsg-name <your-nic-nsg> \
  --resource-group <your-rg> \
  --name Allow-Web-8080 \
  --priority 110 \
  --access Allow \
  --protocol Tcp \
  --destination-port-ranges 8080 \
  --source-address-prefixes <your-ip> \
  --direction Inbound

# Subnet-level NSG (if present)
az network nsg rule create \
  --nsg-name <your-subnet-nsg> \
  --resource-group <your-rg> \
  --name AllowWebUI \
  --priority 1030 \
  --access Allow \
  --protocol Tcp \
  --destination-port-ranges 8080 \
  --source-address-prefixes <your-ip> \
  --direction Inbound
```

## Step 12: Attach a Public IP (if needed)

```bash
# Check if your VM NIC has a public IP
az network nic show -g <your-rg> -n <your-vm-nic> \
  --query "ipConfigurations[0].publicIPAddress.id"

# If null, create and attach one
az network public-ip create -g <your-rg> -n <your-vm>-pip --sku Standard
az network nic ip-config update -g <your-rg> \
  --nic-name <your-vm-nic> \
  --name <ip-config-name> \
  --public-ip-address <your-vm>-pip
```

Access the web UI at `http://<public-ip>:8080`.

## Running Services Summary

| Service | Port | Command |
|---------|------|---------|
| vLLM (Qwen3.5-27B + DFlash) | 8000 | `vllm serve Qwen/Qwen3.5-27B --spec-model z-lab/Qwen3.5-27B-DFlash ...` |
| Web app (FastAPI) | 8080 | `python api.py --config my.yaml --host localhost --port 8080` |
| Cosmos DB | 443 | Azure cloud (no local process) |
| Azure OpenAI (Original backend) | 443 | Azure cloud (no local process) |

## Disk Space Requirements

| Item | Size |
|------|------|
| Qwen3.5-27B weights | ~52 GB |
| Qwen3.5-27B-DFlash draft model | ~1 GB |
| Qwen3-Embedding-0.6B | ~1.2 GB |
| vLLM compile cache | ~3 GB |
| Python venv + packages | ~15 GB |
| OS + tools | ~20 GB |
| **Total minimum** | **~100 GB** (recommend 256GB+) |

## Troubleshooting

### vLLM won't start

- **`fp8_e5m2 kv-cache is not supported with fp8 checkpoints`**: Remove `--kv-cache-dtype fp8_e5m2` from the vLLM command. This was deprecated in vLLM 0.23.0.
- **`unrecognized arguments: --speculative-model`**: Use `--spec-model` and `--spec-tokens` instead. The argument names changed in vLLM 0.23.0.
- **Out of memory**: Reduce `--gpu-memory-utilization` to 0.85 or decrease `--max-model-len` to 8192.

### Web app "Connection error" on DFlash/KG queries

vLLM is not running on port 8000. Start it first (Step 9) and wait for `Application startup complete`.

### Cosmos DB 403 Forbidden

Your Azure identity lacks the required RBAC roles. See Step 7 for the roles to request from your admin.

### Semantic reranker returns 403

The "Semantic Reranker User" role must be assigned on the **InferenceService** scope (`Microsoft.InferenceService/inferenceAccounts/divdet`), not the DocumentDB scope. See Step 7.

### Can't reach the web UI from browser

1. Check your current IP matches the NSG rule: `curl ifconfig.me`
2. Verify both NIC-level and subnet-level NSGs have the rule
3. Verify the public IP is attached to the VM NIC
