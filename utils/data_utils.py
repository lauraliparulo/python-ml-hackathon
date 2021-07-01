import dask.dataframe as dd
import pandas as pd 
import json
from pandas import json_normalize
from timeit import default_timer as timer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import csv
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
#report dependencies
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import joblib
from utils.unsorted_label_encoder import UnsortedLabelEncoder

header_list = ["id", "versicherungsnummer", "vorname","nachname", "geburtsdatum", "ort",
               "strasse", "telefon", "iban","email", "emaildatum", "kategorie", "betreffzeile"]

def data_set_from_dir(data_set_dir_path):
    dataFrame = data_ingestion(data_set_dir_path)
    return data_set_test_preparation_from_dataframe(dataFrame)

def data_set_from_dir_test(data_set_dir_path):
    dataFrame = data_ingestion(data_set_dir_path)
    return data_set_test_preparation_from_dataframe_test(dataFrame)

def data_set_test_preparation_from_dataframe(dataframe):
    dataframe = data_cleansing(dataframe)
    subjectsXtrain, subjectsXtest, categoriesYtrain, categoriesYtest = data_splitting_and_vectorizing(dataframe)
    return subjectsXtrain, subjectsXtest, categoriesYtrain, categoriesYtest

def data_set_test_preparation_from_dataframe_test(dataframe):
    dataframe = data_cleansing(dataframe)
    return data_vectorizing(dataframe)

def json_string_to_data_set(json_string):
    json_object = json.loads(json_string)
    print(json_object) 
    dataframe = json_normalize(json_object['datasets'])   
    print(dataframe)
    return dataframe 

def json_string_to_data_set(json_string):
    json_object = json.loads(json_string)
    print(json_object) 
    dataframe = json_normalize(json_object['datasets'])   
    print(dataframe)
    return dataframe 

def data_set_test_preparation_upload(data_set_dir_path):
    dataFrame = data_ingestion_upload(data_set_dir_path)
    dataFrame = data_cleansing(dataFrame)
    if not isinstance(dataFrame, pd.DataFrame):
         df = dataFrame.compute()
         subjectsTest, categoriesTest = data_vectorizing(df)         
    else: 
         subjectsTest, categoriesTest =  data_vectorizing(dataFrame)         
   
    labels = list(dataFrame.kategorie.unique())
    labels.append('None')
      
    return subjectsTest, categoriesTest, labels

def data_ingestion(data_set_dir_path):
    print("\nDATA INGESTION started...")
    dataFrame = dd.read_csv(data_set_dir_path+"/*.csv", error_bad_lines=False, delimiter=';',skiprows=1, names=header_list , dtype={'strasse':str})
    print("\nDATA INGESTION completed")
    return dataFrame

def data_ingestion_upload(data_set_dir_path):
    print("\nDATA INGESTION started...")
    dataFrame = dd.read_csv(data_set_dir_path, error_bad_lines=False, delimiter=';',skiprows=1, names=header_list , dtype={'strasse':str})
    print(dataFrame.head())
    print("\nDATA INGESTION completed")
    return dataFrame

def data_cleansing(dataFrame):    
    dataFrame.drop(dataFrame.loc[:, 'versicherungsnummer':'emaildatum'].columns, axis = 1)
    print(dataFrame.head())
    print ("\nDATA CLEANSING started...")
    starttime = timer()

    dataFrame = dataFrame[dataFrame['kategorie'].notnull()]
    dataFrame = dataFrame[dataFrame['kategorie']!=""]
    dataFrame = dataFrame[dataFrame['betreffzeile'].notnull()]
    dataFrame = dataFrame[dataFrame['betreffzeile']!=""]

    print("Time to remove nulls", timer() - starttime)
    labels = list(dataFrame.kategorie.unique())
    print("Categories found: ",labels)
    print ("\nDATA CLEANSING completed")
    return dataFrame

def data_splitting_and_vectorizing(dataFrame):
    print ("\nSPLITTING DATA started...")
    
    if not isinstance(dataFrame, pd.DataFrame):
        dataFrameP = dataFrame.compute();
    else: dataFrameP = dataFrame
    
    subject_train, subject_test, categories_train, categories_test = train_test_split(dataFrameP.betreffzeile, dataFrameP.kategorie, test_size=0.25, random_state=10)
    #trainingDataframe, testDataFrame = dataFrame.random_split([0.85, 0.15], random_state=123)
    print ("\nSPLITTING DATA completed")

    print("\nVECTORIZING subjects and categories...")
    vectorizer = CountVectorizer(max_features=100, min_df=5, max_df=0.8, stop_words=stopwords.words('german'))
    subjectsXtrain = vectorizer.fit_transform(subject_train)
    subjectsXtest = vectorizer.fit_transform(subject_test)

    print('vect.get_feature_names(): {0}'.format(vectorizer.get_feature_names()))
   
    filename = 'trained_models/count_vector.sav'
    joblib.dump(vectorizer, filename)

    encoder = UnsortedLabelEncoder()
    encoder.fit(categories_train)
    print('get labels: {0}'.format( encoder._get_param_names()))
    categoriesYtrain = encoder.transform(categories_train)
    categoriesYtest = encoder.transform(categories_test)
  
   
    filename = 'trained_models/label_encoder.sav'
    joblib.dump(encoder, filename)
  
    return subjectsXtrain, subjectsXtest, categoriesYtrain, categoriesYtest

   

def data_vectorizing(dataFrame):
    print("\nVECTORIZING subjects and categories...")
    vectorizer = CountVectorizer(max_features=100, min_df=5, max_df=0.8, stop_words=stopwords.words('german'))
    subjectsTest = vectorizer.fit_transform(dataFrame.betreffzeile)
    saveCountVectorizer(vectorizer)
    encoder = LabelEncoder()
    categoriesTest = encoder.fit_transform(dataFrame.kategorie)
    print("\nVECTORIZING completed")
   
    return subjectsTest, categoriesTest 

def data_preparation_eval(dataframe):
    print('Eval prep')
    dataframe.drop(dataframe.loc[:, 'versicherungsnummer':'emaildatum'].columns, axis = 1)
    dataframe = data_cleansing(dataframe)
    print("HEAD: ",dataframe.head())
    subjectsTest, categoriesTest, labels = data_vectorizing(dataframe)
    return subjectsTest, categoriesTest

def report_classification(categoriesTest, categoriesPred):
    print("\nREPORT results")
    conf_matrix =  confusion_matrix(categoriesTest, categoriesPred)
    report= classification_report(categoriesTest, categoriesPred)
    print(report)
    print(conf_matrix)
    accuracy = accuracy_score(categoriesTest, categoriesPred)
    print("accuracy:   %0.3f" % accuracy)

    print("---------------------------")
    return conf_matrix, report, accuracy
    
def csv_to_json(csvFilePath):
    jsonArray = []
    #read csv file
    with open(csvFilePath, encoding='utf-8') as csvf: 
        #load csv file data using csv library's dictionary reader
        csvReader = csv.DictReader(csvf) 
        #convert each csv row into python dict
        for row in csvReader: 
            #add this python dict to json array
            jsonArray.append(row)
            
        jsonString = json.dumps(jsonArray, indent=4)
    return jsonString
    