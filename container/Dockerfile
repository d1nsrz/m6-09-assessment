FROM python:3.11-slim

# Keeps Python from buffering stdout/stderr so logs appear immediately
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (layer-caching friendly)
COPY container/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code, model weights, and identity file
COPY container/app        /app/app
COPY container/models     /app/models
COPY container/STUDENT.json /app/STUDENT.json

# Make the package importable as `app.*`
ENV PYTHONPATH=/app

ENTRYPOINT ["python", "/app/app/cli.py"]
