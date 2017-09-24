import csv

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline

class RemaxDataset:
    def __init__(self):
        self.id =[]
        self.data = []
        self.nrquartos = []
        self.andar = []
        self.casasbanho = []
        self.assoalhadas = []

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def parsedata():
    """ Returns dictionary with label -> description """
    csvfilepath = '/Users/helioteixeira/Library/Mobile Documents/com~apple~CloudDocs/DataAnalytics/DadosRemax/Remax data_revised_UTF8.csv'
    with open(csvfilepath , 'rt', encoding='UTF8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';')

        remaxdata_labeled = RemaxDataset()
        remaxdata_unlabeled = RemaxDataset()

        next(csvreader, None) 
        for csvreader_row in csvreader:
            if (csvreader_row[11]!=''):
                remaxdata_labeled.id.append(int(csvreader_row[0]))
                remaxdata_labeled.data.append(csvreader_row[3]) 
                remaxdata_labeled.nrquartos.append(csvreader_row[8])
                remaxdata_labeled.andar.append(csvreader_row[13])
                remaxdata_labeled.casasbanho.append(csvreader_row[9])
                remaxdata_labeled.assoalhadas.append(csvreader_row[11])
            else:
                remaxdata_unlabeled.id.append(int(csvreader_row[0]))
                remaxdata_unlabeled.data.append(csvreader_row[3]) 
                
        return (remaxdata_labeled, remaxdata_unlabeled)

def tokenizar_palavras(texto):
    portuguese_tokenizer = nltk.data.load('tokenizers/punkt/PY3/portuguese.pickle')    
    frases = portuguese_tokenizer.tokenize(texto)
    words_list = []
    return [word for frase in frases for word in nltk.word_tokenize(frase)]

def bag_of_words(words):
    return dict((word, True) for word in words)

def bag_of_words_not_in_set(words, badwords):
    return bag_of_words(set(words)-set(badwords))

def bag_of_non_stopwords(words, stopfile='portuguese'):
    badwords = stopwords.words(stopfile)
    return bag_of_words_not_in_set(words, badwords)

def bag_of_bigram_words(words, score_fn=BigramAssocMeasures.chi_sq, n=100):
    bigram_finder= BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_non_stopwords(words+bigrams)

def slice_train_test(dataset):
    size = int(len(dataset) * 0.9)

    train = dataset[:size]
    test = dataset[size:]

    return (train, test)


"""
tokenizar words
criar bag of non stop words e bigrams
"""

def main():
    """ Programa principal """
    #nltk.download('stopwords')
    (remax_labeled , remax_unabeled) = parsedata()

    size = int(len(remax_labeled.data) * 0.9)

    traindata = remax_labeled.data[:size]
    trainlabel = remax_labeled.assoalhadas[:size]

    testdata = remax_labeled.data[size:]
    testlabel = remax_labeled.assoalhadas[size:]

    # BoW
    count_vect = CountVectorizer(encoding=u'utf-8', lowercase=True, analyzer=u'word')
    X_train_data = count_vect.fit_transform(traindata)

    # TFiDF
    tfidf_transformer = TfidfTransformer()
    X_train_data_tfidf = tfidf_transformer.fit_transform(X_train_data)

    # Create Naive Bayes Classifier
    classifier_nb = MultinomialNB().fit(X_train_data_tfidf , trainlabel)

    # Create SVM Classifier
    classifier_svm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42).fit(X_train_data_tfidf , trainlabel)

    # Predict a class for each description
    X_test_data = count_vect.transform(testdata)
    X_test_data_tfidf = tfidf_transformer.transform(X_test_data)

    predicted_nb = classifier_nb.predict(X_test_data_tfidf)
    predicted_svm = classifier_svm.predict(X_test_data_tfidf)

    print('Naive Bayes Classification:')
    print(metrics.classification_report(
         testlabel,
         predicted_nb))

    print('SVM Classification:')
    print(metrics.classification_report(
         testlabel,
         predicted_svm))


if __name__ == "__main__":
    main()
