from flask import Flask
from flask import render_template, request, make_response, jsonify, json, redirect, url_for
from os import remove
from data_utils import data_set_test_preparation_upload
from data_utils import json_string_to_data_set
from training_models_utils import score_with_RandomForest
from training_models_utils import score_with_LogisticRegression
from training_models_utils import score_with_LinearSVC
from data_utils import json_string_to_data_set
from data_utils import data_set_test_preparation_from_dataframe_test
from data_utils import data_ingestion
from pandas import json_normalize
import pandas as pd
import numpy as np
import dask as dd
from werkzeug.utils import secure_filename
from data_utils import csv_to_json
import os
from json import JSONEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

app = Flask(__name__)

algorithms = ['linearSVC','randomForest','logisticRegression']

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

#upload from file by curl
@app.route('/upload_json', methods = ['POST','PUT'])
def upload_dataset_json():
    
      json_data = request.get_json()
      #print(json_data)
      try:
           algorithm = json_data["algorithm"]
      except KeyError:
          print("algorithm not found in JSON. Using logistic regression")
          algorithm = algorithms[2]
      # dataframe = json_normalize(json_data['datasets'])     
      # print(dataframe)     
      
      ##  --- TODO remove after ------------------------------------------------------------
      # restore for calls
      dataframe = data_ingestion("people-csv-light")
      # ------------------------------------------------------------------------------------
      
      subjectsTest, categoriesTest = data_set_test_preparation_from_dataframe_test(dataframe)
  
      categories_labels = dataframe.kategorie.dropna().unique();

      labels = list(categories_labels)
      labels.append('None')
    
      if algorithm == algorithms[0]:
        categoriesPredicted, matrix, report, accuracy = score_with_LinearSVC(subjectsTest, categoriesTest)    
      elif  algorithm == algorithms[1]:
        categoriesPredicted, matrix, report, accuracy = score_with_RandomForest(subjectsTest, categoriesTest)  
      else :
         categoriesPredicted, matrix, report, accuracy = score_with_LogisticRegression(subjectsTest, categoriesTest)  
     
      report = classification_report(categoriesTest, categoriesPredicted, target_names=list(labels))
       
      report_df = classification_report_to_dataframe(report)
      gruppen_id = "Capgemini Springboot Team"
      id ="dummy-id-4711"   
      
      print("report:")
      print(report)
      
      kategories = []
      precisions = []
      recall = []
      f1_score = []
      support = []
      
      i=0
      
      for row in report_df.iterrows(): 
            precisions.append(row[1]['precision'])   
            recall.append(row[1]['recall'])
            f1_score.append(row[1]['f1-score'])  
            support.append(row[1]['support'])    

      for label in labels: 
          kategories.append({'Kategorie': label, 'Precision': precisions[i], 'Recall':recall[i], 'F1-score':f1_score[i], 'Support':support[i]})  
          i+=1
   
        
      responseBody = JSONEncoder().encode({
          "Responses" : [
              { "gruppen_id" : gruppen_id,  
                "id" : id,
                "algorithm": algorithm,
                "accuracy" : str(accuracy),
                "kategorien" : kategories
               }
              ]})
                 
      return make_response(responseBody,200);  

def classification_report_to_dataframe(str_representation_of_report):
    split_string = [x.split(' ') for x in str_representation_of_report.split('\n')]
    column_names = ['']+[x for x in split_string[0] if x!='']
    values = []
    for table_row in split_string[1:-1]:
        table_row = [value for value in table_row if value!='']
        if table_row!=[]:
            values.append(table_row)
    for i in values:
        for j in range(len(i)):
            if i[1] == 'avg':
                i[0:2] = [' '.join(i[0:2])]
            if len(i) == 3:
                i.insert(1,np.nan)
                i.insert(2, np.nan)
            else:
                pass
    report_to_df = pd.DataFrame(data=values, columns=column_names)
    return report_to_df

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
