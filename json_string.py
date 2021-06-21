from data_utils import data_set_test_preparation
from training_models_utils import score_with_LogisticRegression
from numpy import vectorize

# 4500021;DB-7958-1543202998;Hans-Uwe;Sager;1969-05-25;80362 Grafenau;Tina-Linke-Platz 6;(04240) 72493;DE57363180978792700307;Sager.Hans-Uwe98@web.de;28 Apr 2018 21:13:12 GMT;Schadensmeldung;Versicherung DB-7958-1543202998 Tablet ist verruÃŸt nach Flambiermissgeschick    

json_string = '{"id":"4500021", "versicherungsnummer":"DB-7958-1543202998", "vorname":"Hans-Uwe", "nachname":"Sager", "geburtsdatum":"1969-05-25", "ort":"80362 Grafenau", "strasse":"Tina-Linke-Platz 6","telefon":"(04240) 72493","iban":"DE57363180978792700307","email":"Sager.Hans-Uwe98@web.de","emaildatum":"28 Apr 2018 21:13:12 GMT","kategorie":"Schadensmeldung","betreffzeile":"Versicherung DB-7958-1543202998 Tablet ist verruÃŸt nach Flambiermissgeschick"}'

header_list = ["id", "versicherungsnummer", "vorname","nachname", "geburtsdatum", "ort",
               "strasse", "telefon", "iban","email", "emaildatum", "kategorie", "betreffzeile"]

subjectsTrain, subjectsTest, categoriesTrain, categoriesTest = data_set_test_preparation("people-csv-light")

#score_with_LogisticRegression(subjectsTest, categoriesTest)

#score_with_LogisticRegression(['das ist eine Kündigung'],['Kündigung'])