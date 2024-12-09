FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
COPY emqxsl-ca.crt .
RUN pip install --no-cache-dir -r requirements.txt
COPY subscriber.py .
CMD ["python", "subscriber.py"]
