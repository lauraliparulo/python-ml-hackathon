from flask import Flask
from flask import render_template, request, make_response, jsonify, json, redirect, url_for, Blueprint
from os import remove
from werkzeug.utils import secure_filename
import os
from sklearn.metrics import accuracy_score
from utils.app_utils import *
from utils.data_utils import *
from sklearn.metrics import classification_report
from utils.training_models_utils import load_classifier_for_LogisticRegression
from utils.training_models_utils import load_classifier_for_RandomForest
from utils.training_models_utils import load_classifier_for_LinearSVC
from utils.app_utils import predict_for_dataset
np.set_printoptions(suppress=True,precision=4)
#np.set_printoptions(suppress=True,precision=4)

app = Flask(__name__)
errors = Blueprint('errors', __name__)

@app.route("/")
@app.route("/index")
@app.route('/form')
def upload_file():
   return render_template('upload-csv.html')
    
#@app.route('/form_upload', methods = ['GET', 'POST'])
def uploader_file_from_form():
   if request.method == 'POST':
      f = request.files['file']
      fileName = secure_filename(f.filename)
      f.save(fileName)   
      algorithm = request.form['options']   
      print(algorithm)
      return redirect(url_for('.scoring', algorithm=algorithm, fileName=fileName))
   if request.method == 'GET':  
      return render_template('upload-csv.html',algorithm="none")
  
#@app.route('/uploading')  
def scoring():  
    algorithm = request.args['algorithm'] 
    fileName = request.args['fileName'] 
    print(algorithm)

    subjectsTest, categoriesTest, labels = data_set_test_preparation_upload(fileName)      
    categoriesPredicted, matrix, report, accuracy = score_with_given_algorithm(subjectsTest, categoriesTest, algorithm)
    report = classification_report(categoriesTest, categoriesPredicted)
      
    os.remove(fileName)

    report_df = classification_report_to_dataframe(report)
    return render_template('report.html', cat=categoriesPredicted, matrix=matrix, report = report_df.to_html(), acc=accuracy, alg = algorithm)

#@app.route("/report")
def report():
    report_df = classification_report_to_dataframe(report)

    return render_template('report.html', report = report_df.to_html())

# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

######################################################  
# JSON RESPONSES

#upload from file by curl
@app.route('/upload', methods = ['POST'])
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
      
      
      return make_response(jsonify(response_body),200);

@app.route('/predict/logistic_regression', methods = ['POST'])
def score_with_logistic_regression():
    
      classifier = load_classifier_for_LogisticRegression()
      
      response_body = predict_for_dataset(classifier, request.get_json())
      
      return make_response(response_body,200);  

@app.route('/predict/linear_svc', methods = ['POST'])
def score_with_linear_svc():
    
      classifier = load_classifier_for_LinearSVC()
      
      response_body = predict_for_dataset(classifier, request.get_json())
      
      return make_response(response_body,200);  
  
@app.route('/predict/random_forest', methods = ['POST'])
def score_with_random_forest():
    
      classifier = load_classifier_for_RandomForest()
      
      response_body = predict_for_dataset(classifier, request.get_json())
      
      return make_response(response_body,200);  


@errors.app_errorhandler(Exception)
def handle_error(error):
    message = [str(x) for x in error.args]
    status_code = 500
    print("\nERROR:",message)
    success = False
    response = {
        'success': success,
        'error': {
            'type': error.__class__.__name__,
            'message': message
        }
    }

    return jsonify(response), status_code

app.register_blueprint(errors)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
