# Module for training classifier
# Execute this module before performing prediction task [i.e. sentiment_analysis_engine.py].

import pickle
import constants
from sklearn.svm import LinearSVC
from common_utilities import CommonUtils as utils
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


class Trainer:

    logging = utils.configure_log(constants.TRAINING_LOG)

    @staticmethod
    def training_engine():
        # ----------------------------------------------------------
        # STEP 1: ACQUIRE TRAINING/TESTING DATA
        # ----------------------------------------------------------
        # Read data for training the classifier. We are reading 2 types of data for different purposes:
        # 1. Training Data: This data is purely used for training classifier and finding optimal parameters for our
        #                   final model.
        # 2. Testing Data: This data is used for testing the accuracy of trained classifier.
        log = Trainer.logging
        config = dict()
        log.info("\n\n******** STEP 1: Read data for training the classifier into python ********\n")
        log.info("Read training data")
        reviews_train = utils.data_acquisition(log, constants.TRAINING_DATA_LOCATION)
        log.info("Read output file for training data")
        reviews_train_output = utils.data_acquisition(log, constants.TRAINING_DATA_OUTPUT_LOCATION)
        log.info("Read testing data")
        reviews_test = utils.data_acquisition(log, constants.TESTING_DATA_LOCATION)
        log.info("Read output file for testing data")
        reviews_test_output = utils.data_acquisition(log, constants.TESTING_DATA_OUTPUT_LOCATION)

        # ----------------------------------------------------------
        # STEP 2: DATA PRE-PROCESSING
        # ----------------------------------------------------------
        # Transform raw data to required format by eliminating punctuations, tags, stopwords etc.
        log.info("\n\n******** STEP 2: Data pre-processing ********\n")
        log.info("Clean training data")
        reviews_train = utils.data_pre_processing(log, reviews_train)
        log.info("Clean testing data")
        reviews_test = utils.data_pre_processing(log, reviews_test)

        # ----------------------------------------------------------
        # STEP 3: VECTORIZATION [FEATURE EXTRACTION]
        # ----------------------------------------------------------
        # Convert data into sparse matrix for further processing.
        # Each column corresponds to an unique word in the corpus.
        # Each row corresponds to one review/entry in our data set.
        # Each row contains 1s and 0s where [1] means that the word in corpus corresponding to that column exists
        # in the sentence else [0].
        log.info("\n\n******** STEP 3: Vectorize data ********\n")
        cv = CountVectorizer(binary=True, ngram_range=constants.NGRAM_RANGE)
        cv.fit(reviews_train)
        config['vector'] = cv
        log.info("Vectorize training data")
        reviews_train_vector = utils.vectorize_data(log, reviews_train, cv)
        log.info("Vectorize testing data")
        reviews_test_vector = utils.vectorize_data(log, reviews_test, cv)

        # ----------------------------------------------------------
        # STEP 4: BUILD CLASSIFIER
        # ----------------------------------------------------------
        # Build classifier for different values of regularization to find out one giving highest accuracy.
        log.info("\n\n******** STEP 4: Figure out regularization parameter to build classifier ********\n")
        C = Trainer.train_classifier(log, training_data_vector=reviews_train_vector,
                                     training_data_output=reviews_train_output)

        # ----------------------------------------------------------
        # STEP 5: TRAIN CLASSIFIER
        # ----------------------------------------------------------
        # Train final model on training set using optimal parameters obtained above
        log.info("\n\n******** STEP 5: Build final model of {} classifier ********\n".format(constants.USE_CLASSIFIER))
        final_model = Trainer.train_final_model(log, regularization_parameter=C,
                                                training_data_vector=reviews_train_vector,
                                                training_data_output=reviews_train_output)
        config['classifier'] = {constants.USE_CLASSIFIER: final_model}
        # ----------------------------------------------------------
        # STEP 6: TEST CLASSIFIER
        # ----------------------------------------------------------
        # Model accuracy on test data
        log.info("\n\n******** STEP 6: Check model accuracy for test data ********\n")
        utils.check_accuracy(log, model=final_model, input_vector=reviews_test_vector,
                             expected_output=reviews_test_output)
        # ----------------------------------------------------------
        # STEP 7: SAVE VECTOR AND MODEL
        # ----------------------------------------------------------
        # Save configurations in config file using serialization
        config_writer = open(constants.CONFIG_FILE, 'wb')
        pickle.dump(config, config_writer)
        config_writer.close()
        log.info("Training task complete. Classifier is ready for use.")

    # Utility to prepare final model of classifier for prediction task
    @staticmethod
    def train_final_model(log, regularization_parameter, training_data_vector, training_data_output,
                          classifier_name=constants.DEFAULT_CLASSIFIER):
        if classifier_name == constants.LOGISTIC_REGRESSION:
            log.info("Classifier type: Logistic Regression")
            final_model = LogisticRegression(C=regularization_parameter, max_iter=constants.MAX_ITER)
            final_model.fit(training_data_vector, training_data_output)
        elif classifier_name == constants.LINEAR_SVC:
            log.info("Classifier type: Linear Support Vector Classifier")
            final_model = LinearSVC(C=regularization_parameter)
            final_model.fit(training_data_vector, training_data_output)
        else:
            raise Exception("Classifier %s NOT SUPPORTED!".format(classifier_name))
        log.info("Final model ready after training.")
        return final_model

    # Utility to train the desired classifier
    @staticmethod
    def train_classifier(log, training_data_vector, training_data_output, classifier_name=constants.DEFAULT_CLASSIFIER,
                         test_size=constants.TEST_SIZE, train_size=constants.TRAIN_SIZE):
        # Splitting training data internally into 2 parts, for -
        #   1. Training the classifier (train_size)
        #   2. Testing the trained classifier to judge its accuracy (test_size)
        log.info("Split {} % of training data for training and {} % for testing".format(train_size*100, test_size*100))
        X_train, X_test, y_train, y_test = train_test_split(training_data_vector, training_data_output,
                                                            test_size=test_size, train_size=train_size)

        # Train the selected classifier, if no classifier is specified then the default one is chosen
        if classifier_name == constants.LOGISTIC_REGRESSION:
            log.info("Train Logistic Regression Classifier")
            optimal_regularization = Trainer.logistic_regression_training(log, X_train, X_test, y_train, y_test)
        elif classifier_name == constants.LINEAR_SVC:
            log.info("Train Support Vector Classifier")
            optimal_regularization = Trainer.linear_svc_training(log, X_train, X_test, y_train, y_test)
        else:
            raise Exception("Classifier %s NOT SUPPORTED!".format(classifier_name))

        # Return optimal regularization parameter.
        log.info("Optimal C having highest accuracy: {}".format(optimal_regularization))
        return optimal_regularization

    # Execute Logistic Regression for different values of regularization [C] and return [C] having highest accuracy
    @staticmethod
    def logistic_regression_training(log, X_train, X_test, y_train, y_test):
        highest_score = 0
        index_of_highest_score = 0
        for index, c in enumerate(constants.REGULARIZATION_RANGE):
            lr = LogisticRegression(C=c, max_iter=constants.MAX_ITER)
            log.info("Fit data")
            lr.fit(X_train, y_train)
            score = utils.check_accuracy(log, model=lr, input_vector=X_test, expected_output=y_test)
            if highest_score < score:
                index_of_highest_score = index
                highest_score = score
            log.info("Accuracy for C = {} ---> {}".format(c, score))
        return constants.REGULARIZATION_RANGE[index_of_highest_score]

    # Execute Support Vector Machine for different values of regularization [C] and return [C] having highest accuracy
    @staticmethod
    def linear_svc_training(log, X_train, X_test, y_train, y_test):
        highest_score = 0
        index_of_highest_score = 0
        for index, c in enumerate(constants.REGULARIZATION_RANGE):
            svm = LinearSVC(C=c, max_iter=constants.MAX_ITER)
            log.info("Fit data")
            svm.fit(X_train, y_train)
            score = utils.check_accuracy(log, model=svm, input_vector=X_test, expected_output=y_test)
            if highest_score < score:
                index_of_highest_score = index
                highest_score = score
            log.info("Accuracy for C = {} ---> {}".format(c, score))
        return constants.REGULARIZATION_RANGE[index_of_highest_score]


if __name__ == "__main__":
    t = Trainer()
    t.training_engine()
