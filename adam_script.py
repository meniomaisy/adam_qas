# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:36:28 2020

@author: elirane
"""

import logging
import os
import sys,getopt
import qas
import pandas
import joblib
import spacy
from time import time
from numpy import array
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

from qas.constants import CORPUS_DIR, EN_MODEL_MD
from qas.corpus.data import QUESTION_CLASSIFICATION_TRAINING_DATA, QUESTION_CLASSIFICATION_MODEL

_logger = logging.getLogger("__main__")
import spacy
from qas.candidate_ans import get_candidate_answers
from qas.classifier.question_classifier import classify_question
from qas.constants import EN_MODEL_MD, EN_MODEL_DEFAULT, EN_MODEL_SM
from qas.doc_search_rank import search_rank
from qas.feature_extractor import extract_features
from qas.query_const import construct_query
from qas.wiki.wiki_search import search_wikipedia


logger = logging.getLogger(__name__)


def pre_process(dta):
    return pandas.get_dummies(dta)


def remove_irrelevant_features(df_question):
    df_question_class = df_question.pop('Class')

    df_question.pop('Question')
    df_question.pop('WH-Bigram')

    return df_question_class


def transform_data_matrix(df_question_train, df_question_predict):

    df_question_train_columns = list(df_question_train.columns)
    df_question_predict_columns = list(df_question_predict.columns)

    df_question_trans_columns = list(set(df_question_train_columns + df_question_predict_columns))

    logger.debug("Union Columns: {0}".format(len(df_question_trans_columns)))

    trans_data_train = {}

    for feature in df_question_trans_columns:
        if feature not in df_question_train:
            trans_data_train[feature] = [0 for i in range(len(df_question_train.index))]
        else:
            trans_data_train[feature] = list(df_question_train[feature])

    df_question_train = pandas.DataFrame(trans_data_train)
    logger.debug("Training data: {0}".format(df_question_train.shape))
    df_question_train = csr_matrix(df_question_train)

    trans_data_predict = {}

    for feature in trans_data_train:
        if feature not in df_question_predict:
            trans_data_predict[feature] = 0
        else:
            trans_data_predict[feature] = list(df_question_predict[feature])  # KeyError

    df_question_predict = pandas.DataFrame(trans_data_predict)
    logger.debug("Target data: {0}".format(df_question_predict.shape))
    df_question_predict = csr_matrix(df_question_predict)

    return df_question_train, df_question_predict


def naive_bayes_classifier(x_train, y, x_predict):
    gnb = GaussianNB()
    gnb.fit(x_train, y)
    prediction = gnb.predict(x_predict)
    return prediction


def support_vector_machine(df_question_train, df_question_class, df_question_predict):
    lin_clf = LinearSVC()
    lin_clf.fit(df_question_train, df_question_class)
    prediction = lin_clf.predict(df_question_predict)
    return prediction, lin_clf


def predict_question_class(question_clf, df_question_predict):
    return question_clf.predict(df_question_predict), question_clf


def load_classifier_model(model_type="linearSVC"):

    # HELP: Not using the persistent classifier. SVC fails when it encounters previously unseen features at training.
    # Refer the comment in query_container

    training_model_path = os.path.join(CORPUS_DIR, QUESTION_CLASSIFICATION_MODEL)

    if model_type == "linearSVC":
        return joblib.load(training_model_path)


def get_question_predict_data(en_doc=None, df_question_test=None):

    if df_question_test is None:
        # currently only supports single sentence classification
        sentence_list = list(en_doc.sents)[0:1]

    else:
        sentence_list = df_question_test["Question"].tolist()

        import spacy
        en_nlp = spacy.load(EN_MODEL_MD)

    question_data_frame = []

    for sentence in sentence_list:

        wh_bi_gram = []
        root_token, wh_pos, wh_nbor_pos, wh_word = [""] * 4

        if df_question_test is not None:
            en_doc = en_nlp(u'' + sentence)
            sentence = list(en_doc.sents)[0]

        for token in sentence:

            if token.tag_ == "WDT" or token.tag_ == "WP" or token.tag_ == "WP$" or token.tag_ == "WRB":
                wh_pos = token.tag_
                wh_word = token.text
                wh_bi_gram.append(token.text)
                wh_bi_gram.append(str(en_doc[token.i + 1]))
                wh_nbor_pos = en_doc[token.i + 1].tag_

            if token.dep_ == "ROOT":
                root_token = token.tag_

        question_data_frame_obj = {'WH': wh_word, 'WH-POS': wh_pos, 'WH-NBOR-POS': wh_nbor_pos, 'Root-POS': root_token}
        question_data_frame.append(question_data_frame_obj)
        logger.debug("WH : {0} | WH-POS : {1} | WH-NBOR-POS : {2} | Root-POS : {3}"
                     .format(wh_word, wh_pos, wh_nbor_pos, root_token))

    df_question = pandas.DataFrame(question_data_frame)

    return df_question


def classify_question(en_doc=None, df_question_train=None, df_question_test=None):
    """ Determine whether this is a who, what, when, where or why question """

    if df_question_train is None:
        training_data_path = os.path.join(CORPUS_DIR, QUESTION_CLASSIFICATION_TRAINING_DATA)
        df_question_train = pandas.read_csv(training_data_path, sep='|', header=0)

    df_question_class = remove_irrelevant_features(df_question_train)

    if df_question_test is None:
        df_question_predict = get_question_predict_data(en_doc=en_doc)
    else:
        df_question_predict = get_question_predict_data(df_question_test=df_question_test)

    df_question_train = pre_process(df_question_train)
    df_question_predict = pre_process(df_question_predict)

    df_question_train, df_question_predict = transform_data_matrix(df_question_train, df_question_predict)

    question_clf = load_classifier_model()

    logger.debug("Classifier: {0}".format(question_clf))

    predicted_class, svc_clf = support_vector_machine(df_question_train, df_question_class, df_question_predict)

    if df_question_test is not None:
        return predicted_class, svc_clf, df_question_class, df_question_train
    else:
        return predicted_class
def get_nlp(language, lite, lang_model=""):
    err_msg = "Language model {0} not found. Please, refer https://spacy.io/usage/models"
    nlp = None

    if not lang_model == "" and not lang_model == "en":

        try:
            nlp = spacy.load(lang_model)
        except ImportError:
            print(err_msg.format(lang_model))
            raise

    elif language == 'en':

        if lite:
            nlp = spacy.load(EN_MODEL_DEFAULT)
        else:

            try:
                nlp = spacy.load(EN_MODEL_MD)
            except (ImportError, OSError):
                print(err_msg.format(EN_MODEL_MD))
                print('Using default language model')
                nlp = get_default_model(EN_MODEL_DEFAULT)

    elif not language == 'en':
        print('Currently only English language is supported. '
              'Please contribute to https://github.com/5hirish/adam_qas to add your language.')
        sys.exit(0)

    return nlp

def insert(df, row):
    insert_loc = df.index.max()

    if pd.isna(insert_loc):
        df.loc[0] = row
    else:
        df.loc[insert_loc + 1] = row

class QasInit:

    nlp = None
    language = "en"
    lang_model = None
    search_depth = 3
    lite = False

    question_doc = None

    question_class = ""
    question_keywords = None
    query = None

    candidate_answers = None

    def __init__(self, language, search_depth, lite, lang_model=""):
        self.language = language
        self.search_depth = search_depth
        self.lite = lite
        self.lang_model = lang_model
        self.nlp = get_nlp(self.language, self.lite, self.lang_model)

    def get_question_doc(self, question):

        self.question_doc = self.nlp(u'' + question)

        return self.question_doc

    def process_question(self, dfOut):

        self.question_class = classify_question(self.question_doc)
        _logger.info("Question Class: {}".format(self.question_class))
        temp=self.question_class.tostring()
        dfentry[0]=""+self.question_class
        
        self.question_keywords = extract_features(self.question_class, self.question_doc)
        _logger.info("Question Features: {}".format(self.question_keywords))
        dfentry[1]=','.join(self.question_keywords)
        

        
        self.query = construct_query(self.question_keywords, self.question_doc)
        _logger.info("Query: {}".format(self.query))
        dfentry[2]="{}".format(self.query)
        insert(dfOut,dfentry)


    def process_answer(self):

        _logger.info("Retrieving {} Wikipedia pages...".format(self.search_depth))
        search_wikipedia(self.question_keywords, self.search_depth)

        # Anaphora Resolution
        wiki_pages = search_rank(self.query)
        _logger.info("Pages retrieved: {}".format(len(wiki_pages)))

        self.candidate_answers, keywords = get_candidate_answers(self.query, wiki_pages, self.nlp)
        _logger.info("Candidate answers ({}):\n{}".format(len(self.candidate_answers), '\n'.join(self.candidate_answers)))

        return " ".join(self.candidate_answers)

def activate(Qlist, dfOut):
    for question in Qlist:
        dfentry = ["NaN","NaN","NaN"]
        qass = QasInit(search_depth=3 ,language=QasInit.language, lite=False, lang_model=QasInit.language)
        qass.get_question_doc(question)
        qass.process_question(dfOut)  
    return(dfOut)

data = array([['Can an affidavit be used in Beit Din?', 10], ['How can I write HTML and send as an email?', 15], ['How do I remove a Facebook app request?', 14], ['How do you grapple in Dead Rising 3?', 1], ['How do you make a binary image in Photoshop?', 1]]) 
df = pd.DataFrame(data)
df.columns = ["question", "num"]

dfentry = ["NaN","NaN","NaN"]
# dfOut = pd.DataFrame(array([["","",""]]))
# dfOut.columns = ["q_class","q_keywords","quary"]
# dfOut = dfOut[:-1]

logging.basicConfig(level=logging.DEBUG)
en_nlp_l = spacy.load(EN_MODEL_MD)


