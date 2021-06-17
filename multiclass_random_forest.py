from utils.data_utils import data_set_test_preparation
from utils.training_models_utils import score_with_RandomForest

subjectsTest, categoriesTest = data_set_test_preparation("people_csv2")
score_with_RandomForest(subjectsTest, categoriesTest)