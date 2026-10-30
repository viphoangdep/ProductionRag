FROM python:3.12-slim
ENV TF_ENABLE_ONEDNN_OPTS=0 TF_CPP_MIN_LOG_LEVEL=3 TOKENIZERS_PARALLELISM=false
WORKDIR /code
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .
COPY src ./src
COPY output ./output
WORKDIR /code/src
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
