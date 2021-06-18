# syntax=docker/dockerfile:1
FROM python:3
MAINTAINER Laura Liparulo <laura.liparulo@capgemini.com>
ARG APP_NAME=python-hackaton-ML
ENV APP_NAME=${APP_NAME}
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python csv_dataframe.py"] #execute trainings
CMD ["python multiclass_random_forest.py"] #execute trainings
CMD ["python webService.py"]
ENTRYPOINT ["python3"]
