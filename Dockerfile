# syntax=docker/dockerfile:1
FROM python:3.8
MAINTAINER Laura Liparulo <laura.liparulo@capgemini.com>
ARG APP_NAME=python-ml-hackathon
ENV APP_NAME=${APP_NAME}
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
COPY . .
EXPOSE 5000
CMD gunicorn -w 4 -b :5000  --timeout 120000  src.app:app
