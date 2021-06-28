from data_utils import data_set_from_dir_test
from training_models_utils import score_with_LinearSVC

subjectsTest, categoriesTest = data_set_from_dir_test("people-csv-light")

categoriesPredicted, matrix, report, accuracy = score_with_LinearSVC(subjectsTest, categoriesTest)