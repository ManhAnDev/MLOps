FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/scripts/requirements.txt
RUN pip install --no-cache-dir -r scripts/requirements.txt
COPY . /app
ENV PYTHONUNBUFFERED=1
