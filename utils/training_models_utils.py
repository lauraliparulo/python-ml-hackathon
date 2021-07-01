from utils.data_utils import data_set_from_dir
from utils.data_utils import report_classification
from sklearn import model_selection
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def prepareData():
    if not os.path.exists('trained_models'):
        os.makedirs('trained_models')    
    return data_set_from_dir("people_csv")

def load_classifier_for_RandomForest() :
    print("\nLOADING random forest classifier...")
    filename = 'trained_models/randomforest_model.sav'
    # load the model from disk
    classifier = joblib.load(filename)
    print("\nLOADED!")
    return classifier

def score_with_RandomForest(subjectsTest, categoriesTest):
    classifier = load_classifier_for_RandomForest()
    print("\nSCORING with random forest classifier...")
    categoriesPredicted = classifier.predict(subjectsTest)
    matrix, report, accuracy = report_classification(categoriesTest, categoriesPredicted)
    return  categoriesPredicted, matrix, report, accuracy

def load_classifier_for_LogisticRegression() :
    print("\nLOADING logistic regression classifier...")
    filename = 'trained_models/logistic_regression_model.sav'
    # load the model from disk
    classifier = joblib.load(filename)
    return classifier

def score_with_LogisticRegression(subjectsTest, categoriesTest):
    classifier = load_classifier_for_LogisticRegression()
    print("\nSCORING with logistic regression classifier...")
    categoriesPredicted = classifier.predict(subjectsTest)
    matrix, report, accuracy = report_classification(categoriesTest, categoriesPredicted)
    return  categoriesPredicted, matrix, report, accuracy
    
def load_classifier_for_LinearSVC() :
    print("\nLOADING linear SVC classifier...")
    filename = 'trained_models/linearSVC_model.sav'
    # load the model from disk
    classifier = joblib.load(filename)
    return classifier

def score_with_LinearSVC(subjectsTest, categoriesTest):
    classifier = load_classifier_for_LinearSVC()
    print("\nSCORING with LinearSVC classifier...")    
    categoriesPredicted = classifier.predict(subjectsTest)
    matrix, report, accuracy = report_classification(categoriesTest, categoriesPredicted)
    print("Categories predicted")
    print(categoriesPredicted)
    return  categoriesPredicted, matrix, report, accuracy


def saveLinearSVCModel (subjectsXtrain, subjectsXtest, categoriesYtrain, categoriesY):
    print("\nTRAINING WITH Linear SVC  classifier...")
    
    svm = LinearSVC()
    classifier = CalibratedClassifierCV(svm) 
 
    classifier.fit(subjectsXtrain,categoriesYtrain)

    # save the model to disk
    filename = 'trained_models/linearSVC_model.sav'
    joblib.dump(classifier, filename)
    print("\nsaved Linear SVC  classifier...")


def saveLogisticRegressionModel (subjectsXtrain, subjectsXtest, categoriesYtrain, categoriesY): 
    print("\nTRAINING WITH LogisticRegression classifier...")

    kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
    classifier = LogisticRegression(solver='liblinear')
    classifier.fit(subjectsXtrain, categoriesYtrain)

    # save the model to disk
    filename = 'trained_models/logistic_regression_model.sav'
    joblib.dump(classifier, filename)
    print("\nsaved LogisticRegression classifier...")

def saveRandomForestClassifier (subjectsXtrain, subjectsXtest, categoriesYtrain, categoriesYtest):
    print("\nTRAINING WITH random forest classifier...")
    classifier = RandomForestClassifier(n_estimators=10)
    classifier.fit(subjectsXtrain,categoriesYtrain)

    # save the model to disk
    filename = 'trained_models/randomforest_model.sav'
    joblib.dump(classifier, filename)
    print("\nsaved random forest classifier...")
    