FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

# Install torch with CUDA FIRST, then vLLM (pip resolves CPU torch otherwise)
RUN pip3 install torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
RUN pip3 install vllm==0.17.1 jsonschema requests pyyaml huggingface_hub

COPY cmd/ /app/cmd/
COPY pkg/ /app/pkg/
COPY schemas/ /app/schemas/
COPY manifests/ /app/manifests/

ENV PYTHONPATH=/app
ENV VLLM_BATCH_INVARIANT=1
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV PYTHONHASHSEED=0

WORKDIR /workspace
CMD ["python3", "/app/cmd/server/main.py"]
