from flask import Flask
from flask import render_template, request, make_response, jsonify, json, redirect, url_for
from flask_restful import reqparse
from os import remove
from data_utils import data_set_test_preparation_upload
from training_models_utils import score_with_RandomForest
from training_models_utils import score_with_LogisticRegression
from training_models_utils import score_with_LinearSVC
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

algorithms = ['linearSVC','randomForest','logisticRegression']

#parser = reqparse.RequestParser()
#parser.add_argument('algorithm', type=str)

@app.route("/")
@app.route("/index")
@app.route('/form')
def upload_file():
   return render_template('upload-csv.html')
    
@app.route('/form_upload', methods = ['GET', 'POST'])
def uploader_file_from_form():
   if request.method == 'POST':
      f = request.files['file']
      fileName = secure_filename(f.filename)
      f.save(fileName)   
      algorithm = request.form['options']   
      return redirect(url_for('.scoring', algorithm=algorithm, fileName=fileName))
   if request.method == 'GET':  
      return render_template('upload-csv.html',algorithm="none")
  
@app.route('/uploading')  
def scoring():  
    algorithm = request.args['algorithm'] 
    fileName = request.args['fileName'] 
    print(algorithm)
    subjectsTest, categoriesTest = data_set_test_preparation_upload(fileName)
    
    if algorithm == algorithms[0]:
        categoriesPredicted, matrix, report, accuracy = score_with_LinearSVC(subjectsTest, categoriesTest)    
    elif  algorithm == algorithms[1]:
        categoriesPredicted, matrix, report, accuracy = score_with_RandomForest(subjectsTest, categoriesTest)  
    elif algorithm == algorithms[2]:
        categoriesPredicted, matrix, report, accuracy = score_with_LogisticRegression(subjectsTest, categoriesTest)  
    os.remove(fileName)
    return render_template('report.html', cat=categoriesPredicted, matrix=matrix, rep = report, acc=accuracy, alg = algorithm)

@app.route("/report")
def report():
   return render_template('report.html')


# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

######################################################  
# JSON RESPONSES

#upload from file by curl
@app.route('/upload', methods = ['POST','PUT'])
def upload_file_from_request():
      f = request.files['file']
      fileName = secure_filename(f.filename)
      f.save(fileName)   
      print(fileName)
      json_data = request.form.get('data')
      print(json_data)
      d = json.loads(json_data)
      algorithm = d["algorithm"]
      subjectsTest, categoriesTest = data_set_test_preparation_upload(fileName)    
      if algorithm == algorithms[0]:
        categoriesPredicted, matrix, report, accuracy = score_with_LinearSVC(subjectsTest, categoriesTest)    
      elif  algorithm == algorithms[1]:
        categoriesPredicted, matrix, report, accuracy = score_with_RandomForest(subjectsTest, categoriesTest)  
      elif algorithm == algorithms[2]:
         categoriesPredicted, matrix, report, accuracy = score_with_LogisticRegression(subjectsTest, categoriesTest)  
      os.remove(fileName)      
      response = app.response_class( 
        response="Accuracy: "+str(accuracy)+"\n",
        status=200,
        mimetype='application/json')
      responseBody = { "Algorithm": algorithm, "Accuracy": str(accuracy)}
      return make_response(jsonify(responseBody),200);
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)

# endpoint call to trigger training with algorithm

# post/put csv file and trigger with new data...
