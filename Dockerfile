FROM python:3.11-slim-bookworm

# No .pyc files in the image, and log lines appear immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# Dependencies first, so this layer is cached when only the code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

# Accepts the standard arguments: --train --input --output --artifacts-dir
ENTRYPOINT ["python", "run_submission.py"]
