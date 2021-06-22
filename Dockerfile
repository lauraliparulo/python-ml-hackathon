# syntax=docker/dockerfile:1
FROM python:3
MAINTAINER Laura Liparulo <laura.liparulo@capgemini.com>
ARG APP_NAME=python-ml-hackathon
ENV APP_NAME=${APP_NAME}
RUN mkdir /code
WORKDIR /code
ADD requirements.txt /code
RUN pip install -r requirements.txt --no-cache-dir
ADD . /code

# dir

CMD ["python csv_dataframe.py"] #execute trainings
CMD ["python multiclass_random_forest.py"] #execute trainings
CMD ["python webService.py"]
ENTRYPOINT ["python3"]
