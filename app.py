#!/usr/bin/env python3
import flask
from flask import Flask, Response
from flask import render_template, request, make_response, jsonify, json, redirect, url_for, Blueprint
from werkzeug.utils import secure_filename
import os
from utils.app_utils import *
from utils.data_utils import *
from sklearn.metrics import classification_report
from utils.training_models_utils import load_classifier_for_LogisticRegression
from utils.training_models_utils import load_classifier_for_RandomForest
from utils.training_models_utils import load_classifier_for_LinearSVC
from utils.app_utils import predict_for_dataset

np.set_printoptions(suppress=True, precision=4)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'CGH2021Sep_X8439578jfdk'

errors = Blueprint('errors', __name__)


# @app.before_request
# def auth_user():
#    secret = request.headers.get('secret')
#    if secret != app.config.get('SECRET_KEY'):
#        flask.abort(401)


@app.route("/")
@app.route("/index")
@app.route('/form')
def upload_file():
    return render_template('upload-csv.html')


@app.route('/form_upload', methods=['GET', 'POST'])
def uploader_file_from_form():
    if request.method == 'POST':
        f = request.files['file']
        fileName = secure_filename(f.filename)
        f.save(fileName)
        algorithm = request.form['options']
        print(algorithm)
        return redirect(url_for('.scoring', algorithm=algorithm, fileName=fileName))
    if request.method == 'GET':
        return render_template('upload-csv.html', algorithm="none")


@app.route('/uploading')
def scoring():
    algorithm = request.args['algorithm']
    fileName = request.args['fileName']
    print(algorithm)

    subjectsTest, categoriesTest, labels = data_set_test_preparation_upload(fileName)
    categoriesPredicted, matrix, report, accuracy = score_with_given_algorithm(subjectsTest, categoriesTest, algorithm)
    report = classification_report(categoriesTest, categoriesPredicted)

    os.remove(fileName)

    report_df = classification_report_to_dataframe(report)
    return render_template('report.html', cat=categoriesPredicted, matrix=matrix, report=report_df.to_html(),
                           acc=accuracy, alg=algorithm)


@app.route("/report")
def report():
    report_df = classification_report_to_dataframe(report)

    return render_template('report.html', report=report_df.to_html())


# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

######################################################  
# JSON RESPONSES

# upload from file by curl
@app.route('/upload', methods=['POST'])
def upload_file_from_request():
    f = request.files['file']
    fileName = secure_filename(f.filename)
    f.save(fileName)

    json_data = request.form.get('data')
    print(json_data)

    d = json.loads(json_data)

    algorithm = d["algorithm"]
    subjectsTest, categoriesTest, labels = data_set_test_preparation_upload(fileName)

    categoriesPredicted, matrix, report, accuracy = score_with_given_algorithm(subjectsTest, categoriesTest, algorithm)

    report = classification_report(categoriesTest, categoriesPredicted)
    response_body = create_response_body_from_report(report, labels, algorithm, accuracy);

    os.remove(fileName)

    return make_response(jsonify(response_body), 200);


@app.route('/api/predict/logistic_regression', methods=['POST'])
def score_with_logistic_regression():
    check_secret_auth()
    classifier = load_classifier_for_LogisticRegression()
    response_body = predict_for_dataset(classifier, request.get_json())
    return make_response(response_body, 200);

@app.route('/api/predict/linear_svc', methods=['POST'])
def score_with_linear_svc():
    check_secret_auth()
    classifier = load_classifier_for_LinearSVC()
    response_body = predict_for_dataset(classifier, request.get_json())
    return make_response(response_body, 200);


@app.route('/api/predict/random_forest', methods=['POST'])
def score_with_random_forest():
    check_secret_auth()
    classifier = load_classifier_for_RandomForest()
    response_body = predict_for_dataset(classifier, request.get_json())
    return make_response(response_body, 200);


def check_secret_auth():
    secret = request.args.get('secret', default='', type=str)
    if secret != app.config.get('SECRET_KEY'):
        flask.abort(401)

@errors.app_errorhandler(Exception)
def handle_error(error):
    if error.code == 401:
        message = "Missing or Wrong secret in header"
    else:
        message = error.description
    response = error.get_response()
    response.data = json.dumps({
        "code": error.code,
        "name": error.name,
        "description": message,
    })
    print(message)
    response.content_type = "application/json"
    return response

app.register_blueprint(errors)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
