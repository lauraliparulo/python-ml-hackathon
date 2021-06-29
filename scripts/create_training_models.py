# -*- coding: UTF-8 -*-
import pandas as pd 
import json
from utils.training_models_utils import saveRandomForestClassifier
from utils.training_models_utils import saveLinearSVCModel
from utils.training_models_utils import saveLogisticRegressionModel
from utils.training_models_utils import prepareData

subjectsTrain, subjectsTest, categoriesTrain, categoriesTest = prepareData()
saveRandomForestClassifier(subjectsTrain, subjectsTest, categoriesTrain, categoriesTest)
saveLinearSVCModel(subjectsTrain, subjectsTest, categoriesTrain, categoriesTest)
saveLogisticRegressionModel(subjectsTrain, subjectsTest, categoriesTrain, categoriesTest)


