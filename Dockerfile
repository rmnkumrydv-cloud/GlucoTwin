# GlucoTwin dashboard + API. Build AFTER prepare/train/test so data/ and artifacts/ exist:
#   docker compose up --build      ->  dashboard http://localhost:8501, API http://localhost:8000/docs
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir -e .
EXPOSE 8000 8501
CMD ["streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0"]
