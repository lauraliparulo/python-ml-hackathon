from json import JSONEncoder
from sklearn.metrics import classification_report
from cherrypy._cprequest import ResponseBody
from utils.training_models_utils import *
import numpy as np
import pandas as pd


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
          "Responses" : [
              { "gruppen_id" : gruppen_id,  
                "id" : id,
                "algorithm": algorithm,
                "accuracy" : str(accuracy),
                "kategorien" : kategories
               }
              ]})
      
      return responseBody