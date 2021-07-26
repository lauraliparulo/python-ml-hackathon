from json import JSONEncoder
import numpy as np
import pandas as pd
from pandas import json_normalize
from utils.data_utils import data_cleansing
import joblib
from utils.training_models_utils import score_with_LinearSVC
from utils.training_models_utils import score_with_RandomForest
from utils.training_models_utils import score_with_LogisticRegression

np.set_printoptions(suppress=True,precision=4)

algorithms = ['linearSVC','randomForest','logisticRegression']

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


def score_with_given_algorithm(subjectsTest, categoriesTest, algorithm):

    if algorithm == algorithms[0]:
        categoriesPredicted, matrix, report, accuracy = score_with_LinearSVC(subjectsTest, categoriesTest)    
    elif  algorithm == algorithms[1]:
        categoriesPredicted, matrix, report, accuracy = score_with_RandomForest(subjectsTest, categoriesTest)  
    else :
        categoriesPredicted, matrix, report, accuracy = score_with_LogisticRegression(subjectsTest, categoriesTest)  
    return categoriesPredicted, matrix, report, accuracy
           
def create_response_body_from_report(report, labels, algorithm, accuracy):
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
          if label !='None':
             kategories.append({'Kategorie': label, 'Precision': precisions[i], 'Recall':recall[i], 'F1-score':f1_score[i], 'Support':support[i]})  
          i+=1
   
        
      responseBody = JSONEncoder().encode({
          "responses" : [
              { "gruppen_id" : gruppen_id,  
                "id" : id,
                "algorithm": algorithm,
                "accuracy" : str(accuracy),
                "kategorien" : kategories
               }
              ]})
      
      return responseBody
  
def create_response_body_from_predictions(predictions,ids,labels):

      responses=[]
      formatter = "{0:.4f}"
      gruppen_id = "Capgemini Springboot Team"
            
      for row in predictions: 
          kategories = []
          print(labels)
          print(row)
          i=0
          j=0
          for label in labels:
               kategories.append({'name': label, 'prozent': formatter.format(row[0][i])}) 
               i+=1
               
          responses.append(  {
               "id" : ids[j],
               "gruppenid" : gruppen_id,
               "kategorien" : kategories
            })  
          j+=1

      responseBody = JSONEncoder().encode({
          "responses" : responses})
      
      return responseBody
  
def predict_for_dataset(classifier, json_data):
          
      dataframe = json_normalize(json_data['datasets'])     
      
      dataframe =  data_cleansing(dataframe)
  
      categories_labels = dataframe.kategorie.dropna().unique();
      labels = list(categories_labels)
      if len(labels) <= 2:
         labels.append('None')
      
      filename = 'trained_models/count_vector.sav'
      # load the model from disk
      vectorizer = joblib.load(filename)

      vectorizer._validate_vocabulary()

      filename = 'trained_models/label_encoder.sav'

      label_encoder = joblib.load(filename)

      predictions = []
      for subject in dataframe.betreffzeile:
          categories_prob = classifier.predict_proba(vectorizer.transform([subject]))
          predictions.append(categories_prob)

      print(predictions)
 
      labels = label_encoder.inverse_transform(classifier.classes_)
      
      response_body =  create_response_body_from_predictions(predictions, dataframe.id, labels);
    
      return response_body