#!/bin/sh
sudo gunicorn --bind=0.0.0.0:8000 --timeout 260000 app:app