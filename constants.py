# Contains list of all the constants used by entire program

import re
from nltk.corpus import stopwords

# Contains list of all the constants used by entire program

import re
from nltk.corpus import stopwords


# -------------------------------------------------------------------------
# File Parameters
# -------------------------------------------------------------------------
# Location of text file containing data for training classifier
TRAINING_DATA_LOCATION = "<File Path>"
# Location of output file for training data
TRAINING_DATA_OUTPUT_LOCATION = "<File Path>"

# Location of text file containing data for testing classifier
TESTING_DATA_LOCATION = "<File Path>"
# Location of output file for testing data
TESTING_DATA_OUTPUT_LOCATION = "<File Path>"

# Actual data to perform prediction task on
DATA_FOR_PREDICTION = "<File Path>"
# Output of prediction task
OUTPUT_OF_PREDICTION = "FinalPrediction.txt"
# Prediction Summary
SUMMARY = "Summary.txt"

# -------------------------------------------------------------------------
# List of supported classifiers
# -------------------------------------------------------------------------
LINEAR_SVC = "LinearSVC"
LOGISTIC_REGRESSION = "LogisticRegression"


# -------------------------------------------------------------------------
# Classifier being used in our engine
# -------------------------------------------------------------------------
USE_CLASSIFIER = LINEAR_SVC  # Allowed values for supported classifiers: [LOGISTIC_REGRESSION, LINEAR_SVC]
DEFAULT_CLASSIFIER = LOGISTIC_REGRESSION


# -------------------------------------------------------------------------
# Training Parameters
# -------------------------------------------------------------------------
TRAIN_MODEL = True
TRAIN_SIZE = 0.75   # 75% of training data is used to train the classifier
TEST_SIZE = 0.25    # 25% of training data is used for testing the classifier
REGULARIZATION_RANGE = [0.01, 0.05, 0.25, 0.5, 1]
MAX_ITER = 10000    # Maximum iterations required for solver to converge


# -------------------------------------------------------------------------
# Data Pre-processing Parameters
# -------------------------------------------------------------------------
NGRAM_RANGE = (1, 3)

# List of common punctuations found in text
PUNCTUATIONS_IN_TEXT = "[.;:!\'?,\"()\[\]]"#'\'|\.|\^|\*|\$|\||\+|;|,|:|!|\?|"|\(|\)|\[|\]|\{|\}'
PUNCTUATIONS_IN_TEXT_OBJECT = re.compile(PUNCTUATIONS_IN_TEXT)

# HTML Tags <><\>, hypenated words (Ex: "x-max" etc), forward/backward slashes
TAGS_IN_TEXT = "(<br\s*/><br\s*/>)|(\-)|(\/)"#"(<\w*\s*/><\w*\s*/>)|(\-)|(\/)"
TAGS_IN_TEXT_OBJECT = re.compile(TAGS_IN_TEXT)

# List of english stop words
LANGUAGE = 'english'
ENGLISH_STOP_WORDS = stopwords.words(LANGUAGE)


# -------------------------------------------------------------------------
# Configurations
# -------------------------------------------------------------------------
TRAINING_LOG = "train_classifier.log"
MAIN_LOG = "sentiment_analysis.log"
CONFIG_FILE = "config"
LOG_LEVEL_MAP = { 1: 'DEBUG',
                  2: 'INFO',
                  3: 'WARN',
                  4: 'ERROR',
                  5: 'CRITICAL' }
SET_LOG_LEVEL = LOG_LEVEL_MAP[2]
