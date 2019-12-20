# Main module for performing predication task

import pickle
import constants
from common_utilities import CommonUtils as utils


class SentimentAnalyzer:

    logging = utils.configure_log(constants.MAIN_LOG)

    def __init__(self):
        log = SentimentAnalyzer.logging
        config_reader = open(constants.CONFIG_FILE, 'rb')
        config = pickle.load(config_reader)
        self.analyzing_engine(log, config)
        config_reader.close()

    def analyzing_engine(self, log, config):
        # ----------------------------------------------------------
        # STEP 1: DATA ACQUISITION
        # ----------------------------------------------------------
        # Acquire data from desired source and convert it to python list.
        # Each entry in list corresponds to one line in input text.
        log.info("\n\n******** STEP 1: Read data from source into python ********\n")
        input_data = utils.data_acquisition(log, input_location=constants.DATA_FOR_PREDICTION)

        # ----------------------------------------------------------
        # STEP 2: DATA PRE-PROCESSING
        # ----------------------------------------------------------
        # Transform raw data to required format by eliminating punctuations, tags, stopwords etc.
        log.info("\n\n******** STEP 2: Data pre-processing: Clean raw data. ********\n")
        input_data = utils.data_pre_processing(log, raw_text=input_data)

        # ----------------------------------------------------------
        # STEP 3: VECTORIZATION [FEATURE EXTRACTION]
        # ----------------------------------------------------------
        # Convert data into sparse matrix for further processing.
        # Each column corresponds to an unique word in the corpus.
        # Each row corresponds to one review/entry in our data set.
        # Each row contains 1s and 0s where [1] means that the word in corpus corresponding to that column exists
        # in the sentence else [0].
        # Example:
        # Text: ["Apple is a fruit.",
        #        "Apple is red."]
        # After data pre-processing, our text becomes:
        # Filtered Text: ["Apple fruit.",
        #                 "Apple red."]
        # Corpus: Apple fruit red
        # Vector: [[ 1    1    0],
        #          [ 1    0    1]]
        log.info("\n\n******** STEP 3: Vectorize data ********\n")
        vector_object = config['vector']
        input_vector = utils.vectorize_data(log, input_data, vector_object)

        # ----------------------------------------------------------
        # STEP 4: DATA CLASSIFICATION
        # ----------------------------------------------------------
        # Trigger data classification and store the prediction results in Summary.txt
        log.info("\n\n******** STEP 4: Data classification using {} ********.\n".format(constants.USE_CLASSIFIER))
        if constants.USE_CLASSIFIER not in config['classifier']:
            raise Exception("{} not trained/supported. Please execute training module for {}.".format(
                constants.USE_CLASSIFIER, constants.USE_CLASSIFIER))
        classifier_model_object = config['classifier'][constants.USE_CLASSIFIER]
        utils.classify_data(log, model=classifier_model_object, input_vector=input_vector, write_output=True,
                            output_location=constants.OUTPUT_OF_PREDICTION, classifier_name=constants.USE_CLASSIFIER)
        log.info("Predication complete!!!")


if __name__ == "__main__":
    inputReader = open(constants.DATA_FOR_PREDICTION, "r")
    SentimentAnalyzer()
