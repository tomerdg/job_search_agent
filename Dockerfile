FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TZ=Asia/Jerusalem
COPY requirements.txt requirements-ml.txt requirements-rag.txt ./
# requirements-ml.txt: the relevance model in shadow mode (pandas, numpy, scipy, scikit-learn).
# requirements-rag.txt: local embeddings for the retrieval index and prompt v3 (fastembed, ONNX, no torch).
RUN pip install --no-cache-dir -r requirements.txt -r requirements-ml.txt -r requirements-rag.txt
# The embedding model is baked into the image, so a run never needs the network to index or score.
ENV RAG_MODEL_CACHE=/opt/fastembed HF_HUB_OFFLINE=1
RUN HF_HUB_OFFLINE=0 python -c "from fastembed import TextEmbedding; \
TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_dir='/opt/fastembed')"
COPY jobagent ./jobagent
COPY config ./config
# The CV and the DuckDB file are mounted at runtime (see the workflow), so the image holds no personal data.
ENTRYPOINT ["python", "-m", "jobagent.main"]
