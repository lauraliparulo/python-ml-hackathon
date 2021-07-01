# -*- coding: UTF-8 -*-
from utils.data_utils import data_set_from_dir
from utils.data_utils import data_ingestion
from utils.data_utils import data_cleansing
from utils.data_utils import data_splitting_and_vectorizing

from utils.training_models_utils import score_with_LogisticRegression
from utils.training_models_utils import load_classifier_for_LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
import joblib
from utils.unsorted_label_encoder import UnsortedLabelEncoder
#subjectsTest, categoriesTest = 
np.set_printoptions(suppress=True,precision=4)
#categoriesPredicted, matrix, report, accuracy = score_with_LogisticRegression(subjectsTest, categoriesTest)

classifier = load_classifier_for_LogisticRegression()
print("\nSCORING with logistic regression classifier...")
#categoriesPredicted = classifier.predict(subjectsTest)
dataframe = data_ingestion("people_csv")

df= data_cleansing(dataframe)

data_splitting_and_vectorizing(df)

filename = 'trained_models/count_vector.sav'
    # load the model from disk
vectorizer = joblib.load(filename)

vectorizer._validate_vocabulary()

filename = 'trained_models/label_encoder.sav'

label_encoder = joblib.load(filename)

#print('loaded_vectorizer.get_feature_names(): {0}'.format(vectorizer.get_feature_names()))

pred  = classifier.predict(vectorizer.transform(["Tarifänderung bei der Wohnungsversicherung BF-5161-2363187462 aufgrund von unvorhergesehenen baulichen Änderungen"]))
pred1 = classifier.predict(vectorizer.transform(["Dobes: Kabelbrand, meine Dachpfanne ist abgebrannt"]))
categories_prob = classifier.predict_proba(vectorizer.transform(["Tarifänderung bei der Wohnungsversicherung BF-5161-2363187462 aufgrund von unvorhergesehenen baulichen Änderungen"]))
categories_prob1 = classifier.predict_proba(vectorizer.transform(["Dobes: Kabelbrand, meine Dachpfanne ist abgebrannt"]))
print(pred) 
print(label_encoder.inverse_transform([pred]))

print(categories_prob)

print(pred1) 
print(label_encoder.inverse_transform(pred1))
print(categories_prob1)

#matrix, report, accuracy = report_classification(categoriesTest, categoriesPredicted)
#return  categoriesPredicted, matrix, report, accuracy

