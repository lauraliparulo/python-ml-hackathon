# -*- coding: UTF-8 -*-
import pandas as pd 
import json
from training_models_utils import saveRandomForestClassifier
from training_models_utils import saveLinearSVCModel
from training_models_utils import saveLogisticRegressionModel
from training_models_utils import prepareData

subjectsXtrain, subjectsXtest, categoriesYtrain, categoriesYtest = prepareData()
saveRandomForestClassifier(subjectsXtrain, subjectsXtest, categoriesYtrain, categoriesYtest)
saveLinearSVCModel(subjectsXtrain, subjectsXtest, categoriesYtrain, categoriesYtest)
saveLogisticRegressionModel(subjectsXtrain, subjectsXtest, categoriesYtrain, categoriesYtest)


