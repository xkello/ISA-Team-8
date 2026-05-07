FROM python:3.11-slim

WORKDIR /app

COPY web-demo/backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY web-demo /app/web-demo

WORKDIR /app/web-demo
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]


