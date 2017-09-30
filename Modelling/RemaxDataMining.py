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
        self.text = []
        self.textheader = ''
        self.label = []
        self.labelheader = ''
    
    def partitiondataset(sizeoftrain=0.9):
        size = int(len(remax_labeled.label) * 0.9)
        traininterval = [0:size]
        testinterval = [size:0]
        
    
    

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def parsecsvdata(csvfilename, id_colindex, text_colindex, label_colindex, has_headers=False):
    """ Returns 2 dictionaries (labeled and unlabeled dataset) with label -> description """
    
    csvfilepath = '/Users/helioteixeira/Library/Mobile Documents/com~apple~CloudDocs/DataAnalytics/EnergyRating_PropertyValue/DataPreparation/Remax data_revised_UTF8.csv'
    
    with open(csvfilepath , 'rt', encoding='UTF8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';')

        remaxdata_labeled = RemaxDataset()
        remaxdata_unlabeled = RemaxDataset()

        if has_headers: next(csvreader, None)
        
        for csvreader_row in csvreader:
            if (csvreader_row[label_colindex]!=''):
                remaxdata_labeled.id.append(int(csvreader_row[id_colindex]))
                remaxdata_labeled.text.append(csvreader_row[text_colindex]) 
                remaxdata_labeled.label.append(csvreader_row[label_colindex])
            else:
                remaxdata_unlabeled.id.append(int(csvreader_row[id_colindex]))
                remaxdata_unlabeled.text.append(csvreader_row[text_colindex]) 
                
        return (remaxdata_labeled, remaxdata_unlabeled)

def tokenizar_palavras(texto):
    portuguese_tokenizer = nltk.description.load('tokenizers/punkt/PY3/portuguese.pickle')    
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
    csvfilename = '/Users/helioteixeira/Library/Mobile Documents/com~apple~CloudDocs/DataAnalytics/EnergyRating_PropertyValue/DataPreparation/Remax data_revised_UTF8.csv'
    (remax_labeled , remax_unabeled) = parsecsvdata(csvfilename, 0, 3, 9, True)

    size = int(len(remax_labeled.label) * 0.9)

    traindata = remax_labeled.text[:size]
    trainlabel = remax_labeled.label[:size]

    testdata = remax_labeled.text[size:]
    testlabel = remax_labeled.label[size:]

    # BoW
    count_vect = CountVectorizer(encoding=u'utf-8', lowercase=True, analyzer=u'word', stop_words=stopwords.words('portuguese'))
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

## 75% Precision para nr quartos
    predictabledata = remax_unabeled.text
    
    # Predict a class for each description
    X_pred_data = count_vect.transform(predictabledata)
    X_pred_data_tfidf = tfidf_transformer.transform(X_pred_data)

    predicted_svm = classifier_svm.predict(X_pred_data_tfidf)

    for (desc, label) in predictabledata, predicted_svm:
        print(predicted_svm + ' : ' + predictabledata)

if __name__ == "__main__":
    main()
