# syntax=docker/dockerfile:1
FROM python:3.8
MAINTAINER Laura Liparulo <laura.liparulo@capgemini.com>
ARG APP_NAME=python-document-service
ENV APP_NAME=${APP_NAME}
# set web server root as working dir
WORKDIR /home/site/wwwroot
COPY requirements.txt .
RUN pip install gunicorn
RUN pip install -r requirements.txt --no-cache-dir
COPY . .
EXPOSE 8000
CMD gunicorn --bind=0.0.0.0:8000 --timeout 260000 app:app
