from data_utils import data_set_test_preparation
from training_models_utils import score_with_RandomForest

subjectsTrain, subjectsTest, categoriesTrain, categoriesTest = data_set_test_preparation("people-csv-light")

score_with_RandomForest(subjectsTest, categoriesTest)