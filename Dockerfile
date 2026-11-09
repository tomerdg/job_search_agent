FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TZ=Asia/Jerusalem
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY jobagent ./jobagent
COPY config ./config
# The CV and the DuckDB file are mounted at runtime (see the workflow), so the image holds no personal data.
ENTRYPOINT ["python", "-m", "jobagent.main"]
