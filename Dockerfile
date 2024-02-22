# Use an official Python runtime as a parent image
FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["python", "app.py"]