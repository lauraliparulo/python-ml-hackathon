# Capgemini Machine learning Hackathon - Python document serv-ce 

## Introduction 
Python machine learning document classification example with email subjects by category  


## Build and Test
- Install Python 3.8 or above.
- Install packages as specified in requirements.txt 
- run app.py

## REST-Endpoints

- POST Request - upload with the HTML form:  
 
    {base_url}/form_upload    (WEB-GUI)

- POST Request - file + algorithm upload : 
      
     {base_url}/upload(see example)
    

- POST Request - JSON UPLOAD  (as specified in the Documentaiton)

		{base_url}/predict/random_forest

       {base_url}/predict/logistic_regression

       {base_url}/predict/linear_svc

### Request Json example

{
    "datasets": [
        {
            "id": "22500076",
            "versicherungsnummer": "BF-5161-2363187462",
            "vorname": "Nelli",
            "nachname": "Hbel",
            "geburtsdatum": "1915-11-24",
            "ort": "44588 Sckingen",
            "strasse": "Jacobi Jckelplatz 18",
            "telefon": "0454238496",
            "iban": "DE11451065720035952811",
            "email": "Hbel_Nelli@hotmail.de",
            "emaildatum": "1 Mar 2019 13:32:53 GMT",
            "kategorie": "Tarifwechsel",
            "betreffzeile": "Tarifnderung bei der Wohnungsversicherung BF-5161-2363187462 aufgrund von unvorhergesehenen baulichen Änderungen"
        },
        {
            "id": "22500076",
            "versicherungsnummer": "BF-5161-2363187462",
            "vorname": "Nelli",
            "nachname": "Höbel",
            "geburtsdatum": "1915-11-24",
            "ort": "44588 Sckingen",
            "strasse": "Jacobi Jckelplatz 18",
            "telefon": "0454238496",
            "iban": "DE11451065720035952811",
            "email": "Hbel_Nelli@hotmail.de",
            "emaildatum": "1 Mar 2019 13:32:53 GMT",
            "kategorie": "Tarifwechsel",
            "betreffzeile": "Meldung Blitzschaden an Schornstein"
        },
        ..
        ]}
        
### Response Json example

{
    "Responses": [
        [
            {
                "id": "dummy-id-4711",
                "gruppen_id": "Capgemini Springboot Team",
                "kategorien": [
                    {
                        "name": "Vertragsänderung",
                        "prozent": 0.0
                    },
                    {
                        "name": "Kündigung",
                        "prozent": 0.0
                    },
                    {
                        "name": "Schadensmeldung",
                        "prozent": 0.0
                    },
                    {
                        "name": "Adressänderungen",
                        "prozent": 0.0
                    },
                    {
                        "name": "Tarifwechsel",
                        "prozent": 1.0
                    },
                    {
                        "name": "Bankverbindungsänderungen",
                        "prozent": 0.0
                    },
                    {
                        "name": "Namensänderung",
                        "prozent": 0.0
                    }
                ]
            },
            {
                "id": "dummy-id-4711",
                "gruppen_id": "Capgemini Springboot Team",
                "kategorien": [
                    {
                        "name": "Vertragsänderung",
                        "prozent": 0.2
                    },
                    {
                        "name": "Kündigung",
                        "prozent": 0.0
                    },
                    {
                        "name": "Schadensmeldung",
                        "prozent": 0.7795043224326631
                    },
                    {
                        "name": "Adressänderungen",
                        "prozent": 0.0
                    },
                    {
                        "name": "Tarifwechsel",
                        "prozent": 0.0
                    },
                    {
                        "name": "Bankverbindungsänderungen",
                        "prozent": 0.0
                    },
                    {
                        "name": "Namensänderung",
                        "prozent": 0.020495677567336855
                    }
                ]
            },
            ...
            ]}
        
    
### Examples

     curl -i -X POST "Content-Type: multipart/mixed" -F file=@Datensatz27.csv -F "data={\"algorithm\":\"linearSVC \"};type=application/json" https://python-ml-hackhaton-apim.azure-api.net:8080/upload
     
     
     
### AZURE WEB APP
          http://python-ml-hackathon.azurewebsites.net:5000/predict/linear_svc
