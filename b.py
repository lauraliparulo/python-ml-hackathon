# -*- coding: UTF-8 -*-
import pandas as pd 
import json
from data_utils import data_preparation_eval
from data_utils import data_vectorizing_one_row
from data_utils import json_string_to_data_set
from training_models_utils import score_with_LogisticRegression
from training_models_utils import score_with_LinearSVC

#jsonData = '{"name": "Frank", "age": 39}'
#jsonToPython = json.loads(jsonData)

json_string = '{"id": "22500076","versicherungsnummer" : "BF-5161-2363187462", "vorname" : "Nelli","nachname" : "H�bel","geburtsdatum" : "1915-11-24",     "ort" : "44588 S�ckingen", "strasse" : "Jacobi J�ckelplatz 18", "telefon" : "0454238496", "iban": "DE11451065720035952811","email" : "H�bel_Nelli@hotmail.de", "emaildatum" : "1 Mar 2019 13:32:53 GMT", "kategorie" : "Tarifwechsel", "betreffzeile" : "Tarif�nderung bei der Wohnungsversicherung BF-5161-2363187462 aufgrund von unvorhergesehenen baulichen �nderungen\"}'

# Creation of the dataframe

dataframe = json_string_to_data_set(json_string)

print("CATEGORIES FOUND: ", dataframe['kategorie'].values)

#dataframe['betreffzeile'], dataframe['kategorie']

subjectsTest, categoriesTest = data_vectorizing_one_row(dataframe)

categoriesPredicted, accuracy = score_with_LinearSVC(subjectsTest, categoriesTest)

#print('category predicted: '+categoriesPredicted)
#print('accuracy: ',accuracy)



