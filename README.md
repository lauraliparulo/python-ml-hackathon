# Introduction 
Python machine learning example  


# Build and Test
TODO: Describe and show how to build your code and run the tests. 

# REST-Endpoints

- POST Request - upload with the HTML form:  /form_upload    (WEB-GUI)

- POST Request - file + algorithm upload (see example)

- POST Request - JSON UPLOAD  {url}/upload_json


# Examples

curl -i -X POST "Content-Type: multipart/mixed" -F file=@Datensatz27.csv -F "data={\"algorithm\":\"linearSVC\"};type=application/json" https://python-ml-hackhaton-apim.azure-api.net:8080/upload




If you want to learn more about creating good readme files then refer the following [guidelines](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-a-readme?view=azure-devops). You can also seek inspiration from the below readme files:
- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore)
