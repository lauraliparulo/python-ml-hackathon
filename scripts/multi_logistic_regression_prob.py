import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from utils.data_utils import data_set_from_dir_test
from utils.training_models_utils import data_set_from_dir

np.set_printoptions(suppress=True,precision=4)

subjectsTrain, subjectsTest, categoriesTrain, categoriesTest = data_set_from_dir("people-csv-light")

#subjectsTest, categoriesTest = data_set_from_dir_test("people-csv-light")

print("\nTRAINING WITH LogisticRegression classifier...")

classifier = LogisticRegression(random_state=0)
classifier.fit(subjectsTrain, categoriesTrain)


categories_prob = classifier.predict_proba(subjectsTest)

print(categories_prob)
# save the model to disk
#filename = 'trained_models/logistic_regression_model.sav'
#joblib.dump(classifier, filename)
#print("\nsaved LogisticRegression classifier...")



#classifier = load_classifier_for_LinearSVC()
#print("\nSCORING with LinearSVC classifier...")    
#categoriesPredicted = classifier.predict(subjectsTest)

#matrix, report, accuracy = report_classification(categoriesTest, categoriesPredicted)
#print("Categories predicted")
#print(categoriesPredicted)
#return  categoriesPredicted, matrix, report, accuracy
