# Module containing list of common utilities for loading data into python, data pre-processing, classification etc.

import logging
import constants
from sklearn.metrics import accuracy_score
from data_preprocessing_engine import DataPreprocessor


class CommonUtils:

    # Utility for log configurations
    @staticmethod
    def configure_log(filename):
        # Create and configure logger
        logging.basicConfig(filename=filename,
                            format='%(asctime)s %(message)s',
                            filemode='w')
        logger = logging.getLogger()
        level = constants.SET_LOG_LEVEL
        if level.upper() == 'DEBUG':
            logger.setLevel(logging.DEBUG)
        elif level.upper() == 'INFO':
            logger.setLevel(logging.INFO)
        elif level.upper() == 'WARN':
            logger.setLevel(logging.WARN)
        elif level.upper() == 'ERROR':
            logger.setLevel(logging.ERROR)
        elif level.upper() == 'CRITICAL':
            logger.setLevel(logging.CRITICAL)
        else:
            raise Exception("Unsupported log level {}!".format(level))
        return logger

    # Utility to acquire data for training (optional) and testing
    @staticmethod
    def data_acquisition(log, input_location):
        log.info("Reading data from : {}".format(input_location))
        data_list = list()
        input_reader = open(input_location, 'r')
        for line in input_reader:
            data_list.append(line.strip())
        log.info("Data successfully loaded in python list")
        return data_list

    # Utility to invoke data pre-processing
    @staticmethod
    def data_pre_processing(log, raw_text):
        log.info("Pre-processing Data")
        return DataPreprocessor.extract_pure_text(log, raw_text)

    # Utility for vectorizing data and feature extraction
    @staticmethod
    def vectorize_data(log, input_data, cv):
        log.info("Vectorize data to obtain sparse matrix")
        input_vector = cv.transform(input_data)
        log.info("Vectorization complete.")
        return input_vector

    # Utility to check accuracy of data prediction
    @staticmethod
    def check_accuracy(log, model, input_vector, expected_output):
        score = accuracy_score(expected_output, model.predict(input_vector))
        log.info("Accuracy: {}".format(score))
        return score

    # Utility to perform data classification using desired classifier and store result in output file and save the
    # summary of predication in summary.txt
    @staticmethod
    def classify_data(log, model, input_vector, write_output=True,
                      output_location=constants.OUTPUT_OF_PREDICTION,
                      classifier_name=constants.USE_CLASSIFIER):
        log.info("Perform data classification using {}".format(classifier_name))
        if classifier_name == constants.LOGISTIC_REGRESSION:
            log.info("Predict using Logistic Regression")
            output = model.predict(input_vector)
        elif classifier_name == constants.LINEAR_SVC:
            log.info("Predict using Linear SVC")
            output = model.predict(input_vector)
        else:
            raise Exception("Classifier %s NOT SUPPORTED!".format(classifier_name))

        # Write output data to file
        if write_output:
            log.info("Write data into file {}".format(write_output))
            output_writer = open(output_location, 'w')
            negative = 0
            positive = 0
            for line in output:
                if int(line.strip()) == 0:
                    negative += 1
                else:
                    positive += 1
                output_writer.write(line)

        # Summary of prediction task
        if negative < positive:
            result = "POSITIVE"
        elif negative > positive:
            result = "NEGATIVE"
        else:
            result = "NEUTRAL"
        total = len(output)

        summary_writer = open(constants.SUMMARY, 'w')
        summary_writer.write("\n =============> PREDICTION SUMMARY <=============\n")
        summary_writer.write("\n Number of Positive Outcomes:  {}".format(positive))
        summary_writer.write("\n Number of Negative Outcomes:  {}".format(negative))
        summary_writer.write("\n --------------------------------------------------")
        summary_writer.write("\n Total Outcomes:               {}\n\n".format(len(output)))
        summary_writer.write("\n Percent of Positive Outcomes: {} %".format((positive*100)/total))
        summary_writer.write("\n Percent of Negative Outcomes: {} %".format((negative*100)/total))
        summary_writer.write("\n --------------------------------------------------")
        summary_writer.write("\n Overall Result:               {}".format(result))
        summary_writer.flush()
        summary_writer.close()
        log.info("Data classification complete. Summary saved successfully.")
