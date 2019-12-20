# ---------------------------------------------------------------------------
# This class contains utilities for data pre-processing.
# Raw input text is passed through following filtering stages:
#   1. Remove punctuations from text and covert it to lower case.
#   2. Replace tags, hypens, slashes by space.
#   3. Eliminate stop words using nltk.
#   <Can be extended to add more stages to optimize space and improve prediction power>
# ---------------------------------------------------------------------------
import constants


class DataPreprocessor:

    # Utility to remove punctuations, html tags, quotations.., stopwords and return pure text.
    @staticmethod
    def extract_pure_text(log, raw_text):

        # FIRST PASS:
        # -----------
        # 1. Convert text to lower case.
        # 2. Delete punctuation symbols from text.
        log.info("===> STAGE 1: Remove punctuations and transform text to lowercase")
        filtered_string = [constants.PUNCTUATIONS_IN_TEXT_OBJECT.sub("", line.lower()) for line in raw_text]
        log.debug("Filtered String after stage 1: \n{}".format(filtered_string))

        # SECOND PASS:
        # -------------
        # 1. Replace HTML Tags <><\>, hypenated words (Ex: "x-max" etc), forward/backward slashes by space.
        log.info("===> STAGE 2: Replace HTML Tags, hypenated words, forward/backward slashes by space")
        filtered_string = [constants.TAGS_IN_TEXT_OBJECT.sub(" ", line.lower()).strip() for line in filtered_string]
        log.debug("Filtered String after stage 2: \n{}".format(filtered_string))

        # THIRD PASS:
        # -----------
        # 1. Eliminate stop words for space optimization.
        log.info("===> STAGE 3: Remove stop words")
        third_pass = list()
        for line in filtered_string:
            third_pass.append(' '.join([word for word in line.split()
                                        if word not in constants.ENGLISH_STOP_WORDS]))
        filtered_string = third_pass
        log.debug("Filtered String after stage 3: \n{}".format(filtered_string))
        log.info("Data Pre-processing complete.")
        return filtered_string
