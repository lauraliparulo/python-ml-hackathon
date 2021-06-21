import dask.dataframe as dd
import pandas as pd 
import json
from timeit import default_timer as timer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
#report dependencies
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

header_list = ["id", "versicherungsnummer", "vorname","nachname", "geburtsdatum", "ort",
               "strasse", "telefon", "iban","email", "emaildatum", "kategorie", "betreffzeile"]

def data_set_test_preparation(data_set_dir_path):
    dataFrame = data_ingestion(data_set_dir_path)
    dataFrame = data_cleansing(dataFrame)
    subjectsXtrain, subjectsXtest, categoriesYtrain, categoriesYtest = data_splitting_and_vectorizing(dataFrame)
    return subjectsXtrain, subjectsXtest, categoriesYtrain, categoriesYtest

def json_string_to_data_set(json_string):

    json_multiple ='['+json_string+','+json_string+','+json_string+']'
    json_object = json.loads(json_multiple)

    dataframe = pd.DataFrame(json_object, index=[0])
    
    print(json_object)
    
    print(dataframe.shape[0])
    
    print("CATEGORIES FOUND: ", dataframe['kategorie'].values)
    dataframe.drop(dataframe.loc[:, 'versicherungsnummer':'emaildatum'].columns, axis = 1)
    dataframe = dataframe[dataframe['kategorie'].notnull()]
    dataframe = dataframe[dataframe['kategorie']!=""]
    dataframe = dataframe[dataframe['betreffzeile'].notnull()]
    dataframe = dataframe[dataframe['betreffzeile']!=""]
    #if null
    return dataframe 
    
def data_set_test_preparation_upload(data_set_dir_path):
    dataFrame = data_ingestion_upload(data_set_dir_path)
    dataFrame = data_cleansing(dataFrame)
    if not isinstance(dataFrame, pd.DataFrame):
        return data_vectorizing(dataFrame.compute())
    else: return data_vectorizing(dataFrame)

def data_ingestion(data_set_dir_path):
    print("\nDATA INGESTION started...")
    dataFrame = dd.read_csv(data_set_dir_path+"/Datensatz*.csv", error_bad_lines=False, delimiter=';',skiprows=1, names=header_list , dtype={'strasse':str})

    dataFrame.drop(dataFrame.loc[:, 'versicherungsnummer':'emaildatum'].columns, axis = 1)

    print(dataFrame.head())

    print("\nDATA INGESTION completed")
    return dataFrame

def data_ingestion_upload(data_set_dir_path):
    print("\nDATA INGESTION started...")
    dataFrame = dd.read_csv(data_set_dir_path, error_bad_lines=False, delimiter=';',skiprows=1, names=header_list , dtype={'strasse':str})

    dataFrame.drop(dataFrame.loc[:, 'versicherungsnummer':'emaildatum'].columns, axis = 1)

    print(dataFrame.head())

    print("\nDATA INGESTION completed")
    return dataFrame

def data_cleansing(dataFrame):    
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
    dataFrameP = dataFrame.compute(); # convert to panda dataFrame
    subject_train, subject_test, categories_train, categories_test = train_test_split(dataFrameP.betreffzeile, dataFrameP.kategorie, test_size=0.25, random_state=10)
    #trainingDataframe, testDataFrame = dataFrame.random_split([0.85, 0.15], random_state=123)
    print ("\nSPLITTING DATA completed")

    print("\nVECTORIZING subjects and categories...")
    vectorizer = CountVectorizer(max_features=100, min_df=5, max_df=0.7, stop_words=stopwords.words('german'))
    subjectsXtrain = vectorizer.fit_transform(subject_train)
    subjectsXtest = vectorizer.fit_transform(subject_test)

    encoder = LabelEncoder()
    categoriesYtrain = encoder.fit_transform(categories_train)
    categoriesYtest = encoder.fit_transform(categories_test)
    
    return subjectsXtrain, subjectsXtest, categoriesYtrain, categoriesYtest

def data_vectorizing(dataFrame):
    print("\nVECTORIZING subjects and categories...")
    vectorizer = CountVectorizer(max_features=100, min_df=5, max_df=0.7, stop_words=stopwords.words('german'))
    subjectsTest = vectorizer.fit_transform(dataFrame.betreffzeile)

    encoder = LabelEncoder()
    categoriesTest = encoder.fit_transform(dataFrame.kategorie)
    print("\nVECTORIZING completed")
    return subjectsTest, categoriesTest  


def data_vectorizing_one_row(dataframe):
    print("\nVECTORIZING subjects and categories...")
    vectorizer = CountVectorizer(max_features=100, min_df=1, max_df=1, stop_words=stopwords.words('german'))
    subjectsTest = vectorizer.fit_transform(dataframe.betreffzeile)

    encoder = LabelEncoder()
    categoriesTest = encoder.fit_transform(dataframe.kategorie)
    print("\nVECTORIZING completed")
    return subjectsTest, categoriesTest  


def data_preparation_eval(dataframe):
    print('Eval prep')
    dataframe.drop(dataframe.loc[:, 'versicherungsnummer':'emaildatum'].columns, axis = 1)
    dataframe = data_cleansing(dataframe)
    print("HEAD: ",dataframe.head())
    subjectsTest, categoriesTest = data_vectorizing(dataframe)
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