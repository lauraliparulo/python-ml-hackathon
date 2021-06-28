# -*- coding: UTF-8 -*-
import pandas as pd 
import json
from data_utils import data_preparation_eval
from data_utils import data_set_test_preparation_from_dataframe_test
from data_utils import json_string_to_data_set
from training_models_utils import score_with_LogisticRegression
from training_models_utils import score_with_LinearSVC

#json_string = '{"datasets":[{"id": "22500076","versicherungsnummer" : "BF-5161-2363187462", "vorname" : "Nelli","nachname" : "H�bel","geburtsdatum" : "1915-11-24",     "ort" : "44588 S�ckingen", "strasse" : "Jacobi J�ckelplatz 18", "telefon" : "0454238496", "iban": "DE11451065720035952811","email" : "H�bel_Nelli@hotmail.de", "emaildatum" : "1 Mar 2019 13:32:53 GMT", "kategorie" : "Tarifwechsel", "betreffzeile" : "Tarif�nderung bei der Wohnungsversicherung BF-5161-2363187462 aufgrund von unvorhergesehenen baulichen �nderungen\"},{"id": "22500076","versicherungsnummer" : "BF-5161-2363187462", "vorname" : "Nelli","nachname" : "H�bel","geburtsdatum" : "1915-11-24",     "ort" : "44588 S�ckingen", "strasse" : "Jacobi J�ckelplatz 18", "telefon" : "0454238496", "iban": "DE11451065720035952811","email" : "H�bel_Nelli@hotmail.de", "emaildatum" : "1 Mar 2019 13:32:53 GMT", "kategorie" : "Tarifwechsel", "betreffzeile" : "Meldung Blitzschaden an Schornstein\"},{"id": "22500076","versicherungsnummer" : "BF-5161-2363187462", "vorname" : "Nelli","nachname" : "H�bel","geburtsdatum" : "1915-11-24",     "ort" : "44588 S�ckingen", "strasse" : "Jacobi J�ckelplatz 18", "telefon" : "0454238496", "iban": "DE11451065720035952811","email" : "H�bel_Nelli@hotmail.de", "emaildatum" : "1 Mar 2019 13:32:53 GMT", "kategorie" : "Tarifwechsel", "betreffzeile" : "Tarif�nderung bei der Wohnungsversicherung BF-5161-2363187462 aufgrund von unvorhergesehenen baulichen �nderungen aber ich muss die Wande reparieren sofort\"}]}'


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
  
    return jsonArray

jsonArray = csv_to_json("people-csv-light/test.csv")

#json_object = json.loads(jsonArray)
print(jsonArray)

#dataframe = json_string_to_data_set(json_datasets)

#once it looks like this you can proceed
#json_string = '{"datasets":[{"id": "22500076","versicherungsnummer" : "BF-5161-2363187462", "vorname" : "Nelli","nachname" : "H�bel","geburtsdatum" : "1915-11-24",     "ort" : "44588 S�ckingen", "strasse" : "Jacobi J�ckelplatz 18", "telefon" : "0454238496", "iban": "DE11451065720035952811","email" : "H�bel_Nelli@hotmail.de", "emaildatum" : "1 Mar 2019 13:32:53 GMT", "kategorie" : "Tarifwechsel", "betreffzeile" : "Tarif�nderung bei der Wohnungsversicherung BF-5161-2363187462 aufgrund von unvorhergesehenen baulichen �nderungen\"},{"id": "22500076","versicherungsnummer" : "BF-5161-2363187462", "vorname" : "Nelli","nachname" : "H�bel","geburtsdatum" : "1915-11-24",     "ort" : "44588 S�ckingen", "strasse" : "Jacobi J�ckelplatz 18", "telefon" : "0454238496", "iban": "DE11451065720035952811","email" : "H�bel_Nelli@hotmail.de", "emaildatum" : "1 Mar 2019 13:32:53 GMT", "kategorie" : "Tarifwechsel", "betreffzeile" : "Meldung Blitzschaden an Schornstein\"},{"id": "22500076","versicherungsnummer" : "BF-5161-2363187462", "vorname" : "Nelli","nachname" : "H�bel","geburtsdatum" : "1915-11-24",     "ort" : "44588 S�ckingen", "strasse" : "Jacobi J�ckelplatz 18", "telefon" : "0454238496", "iban": "DE11451065720035952811","email" : "H�bel_Nelli@hotmail.de", "emaildatum" : "1 Mar 2019 13:32:53 GMT", "kategorie" : "Tarifwechsel", "betreffzeile" : "Tarif�nderung bei der Wohnungsversicherung BF-5161-2363187462 aufgrund von unvorhergesehenen baulichen �nderungen aber ich muss die Wande reparieren sofort\"}]}'

#nest jsons

# json_string = '{"datasets":[{"id": "22500076","versicherungsnummer" : "BF-5161-2363187462", "vorname" : "Nelli","nachname" : "H�bel","geburtsdatum" : "1915-11-24",     "ort" : "44588 S�ckingen", "strasse" : "Jacobi J�ckelplatz 18", "telefon" : "0454238496", "iban": "DE11451065720035952811","email" : "H�bel_Nelli@hotmail.de", "emaildatum" : "1 Mar 2019 13:32:53 GMT", "kategorie" : "Tarifwechsel", "betreffzeile" : "Tarif�nderung bei der Wohnungsversicherung BF-5161-2363187462 aufgrund von unvorhergesehenen baulichen �nderungen\"},{"id": "22500076","versicherungsnummer" : "BF-5161-2363187462", "vorname" : "Nelli","nachname" : "H�bel","geburtsdatum" : "1915-11-24",     "ort" : "44588 S�ckingen", "strasse" : "Jacobi J�ckelplatz 18", "telefon" : "0454238496", "iban": "DE11451065720035952811","email" : "H�bel_Nelli@hotmail.de", "emaildatum" : "1 Mar 2019 13:32:53 GMT", "kategorie" : "Tarifwechsel", "betreffzeile" : "Meldung Blitzschaden an Schornstein\"},{"id": "22500076","versicherungsnummer" : "BF-5161-2363187462", "vorname" : "Nelli","nachname" : "H�bel","geburtsdatum" : "1915-11-24",     "ort" : "44588 S�ckingen", "strasse" : "Jacobi J�ckelplatz 18", "telefon" : "0454238496", "iban": "DE11451065720035952811","email" : "H�bel_Nelli@hotmail.de", "emaildatum" : "1 Mar 2019 13:32:53 GMT", "kategorie" : "Tarifwechsel", "betreffzeile" : "Tarif�nderung bei der Wohnungsversicherung BF-5161-2363187462 aufgrund von unvorhergesehenen baulichen �nderungen aber ich muss die Wande reparieren sofort\"}]}'

# dataframe = json_string_to_data_set(json_string)

# print(dataframe)
# print("CATEGORIES FOUND: ", dataframe['kategorie'].values)
    
subjectsTest, categoriesTest = data_set_test_preparation_from_dataframe_test(dataframe)

categoriesPredicted, accuracy = score_with_LinearSVC(subjectsTest, categoriesTest)

categoriesPredicted, accuracy =  score_with_RandomForest(subjectsTest, categoriesTest)
