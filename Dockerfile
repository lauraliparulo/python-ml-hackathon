# syntax=docker/dockerfile:1
FROM python:3
MAINTAINER Laura Liparulo <laura.liparulo@capgemini.com>
ARG APP_NAME=python-ml-hackathon
ENV APP_NAME=${APP_NAME}
RUN mkdir /code
WORKDIR /code
COPY . /code
ADD requirements.txt /code
RUN pip install -r requirements.txt --no-cache-dir
CMD ["python app.py"]
EXPOSE 5000
ENTRYPOINT ["python3"]
