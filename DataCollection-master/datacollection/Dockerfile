FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir pandas Pillow numpy

RUN mkdir -p /workspace/data

COPY main.py /app/main.py

CMD ["python", "/app/main.py"]
