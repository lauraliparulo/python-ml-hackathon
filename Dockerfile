# syntax=docker/dockerfile:1
FROM python:3
MAINTAINER Laura Liparulo <laura.liparulo@capgemini.com>
ARG APP_NAME=python-document-service
ENV APP_NAME=${APP_NAME}
RUN mkdir /code
WORKDIR /code
ADD requirements.txt /code
RUN pip install -r requirements.txt --no-cache-dir
ADD . /code
EXPOSE 8000
ENTRYPOINT ["sh", "code/gunicorn-startup.sh"]
