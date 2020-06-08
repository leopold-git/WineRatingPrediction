import string

import nltk
import pandas as pd
import re
import random
import collections
from nltk.metrics import (precision, recall, f_measure)

# GOAL: TAKE WINE FEATURES INCL DESCRIPTION AND PREDICT THE WINE RATING

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

data = pd.read_csv("winemag-data-130k-v2.csv")


# features to keep: price, description, variety,
# get rid of country -- 42% US not really good indicator, designation, province, taster name, title
# label: points

# delete unwanted columns/feats


data = data.drop(columns=['country', 'designation', 'region_2','title' , 'taster_name',
       'taster_twitter_handle', 'winery', 'province', 'Unnamed: 0'])

data = data.sample(frac=1)

praiseWords = ["beautiful", "gorgeous", "elegant", "superb", "elegance", "beauty"]
# find text features
def document_features(document):
    document_words = set(document)
    praise = set(praiseWords)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    for word in praise:
        features['praise({})'.format(word)] = (word in document_words)
    return features



text = ""  # keep track of all words across dataset

descriptions = (data['description'][:9000]).tolist()
labels = (data['points'][:9000]).tolist()
prices = (data['price'][:9000]).tolist()





text = ""
for row in descriptions:
    text += row
txt = "".join(c for c in text if not c in string.punctuation)
tokens = nltk.word_tokenize(txt)
tokens = [w.lower() for w in tokens]
tokens = [word for word in tokens if word.isalpha()]


all_words = nltk.FreqDist(tokens)
word_features = list(all_words)[4:2005]
#print(word_features)

labeledData = []

# take distribution of points into account

for i in range(len(descriptions)):
    label = 0
    if labels[i] < 84:
        label = 1
    elif labels[i] >= 84 and labels[i] < 88:
        label = 2
    elif labels[i] >= 88 and labels[i] < 92:
        label = 3
    elif labels[i] >= 92 and labels[i] < 96:
        label = 4
    else:
        label = 5
    price = 0
    if prices[i] > 80:
        price = 3
    elif prices[i] > 60 and prices[i] < 80:
        price = 2
    elif prices[i] > 40 and prices[i] < 60:
        price = 1
    else:
        price = 0
    labeledData.append((descriptions[i], price, label))
random.shuffle(labeledData)
#print(labeledData[:1600])
featuresets = []
for (d,p,c) in labeledData:
   # print(d,c)
    docToken = nltk.word_tokenize(d)
    feat = document_features(docToken)
    feat['price:'] = p
    featuresets.append((feat, c))




num_folds = 10
subset_size = int(round(len(featuresets)/num_folds))

# for the Bayes model
foldAccuracies = []

for i in range(num_folds):
    cv_test = featuresets[i*subset_size:][:subset_size]
    cv_train = featuresets[:i*subset_size] + featuresets[(i+1)*subset_size:]
    # use NB classifier
    classifier = nltk.NaiveBayesClassifier.train(cv_train)
    print('  ')
    print('FOLD ' + str(i))
    print('For this fold:')
    print('Accuracy on Fold Test Set: ' + str(nltk.classify.accuracy(classifier, cv_test)))
    foldAccuracies.append(str(nltk.classify.accuracy(classifier, cv_test)));
    # most informative feauures


    for i, (feats, label) in enumerate(cv_test):
        observed = classifier.classify(feats)



    # most informative feauures
    # now get fold stats such as precison, recall, f score




    classifier.show_most_informative_features(10)

total = 0
totalPrecPos = 0
totalRecallPos = 0
totalFScorePos = 0
totalPrecNeg = 0
totalRecallNeg = 0
totalFScoreNeg = 0
for i in range(0, len(foldAccuracies)):
    total = total + float(foldAccuracies[i])



total_accuracy = total / num_folds


print('---------')
print('Averaged model performance over 10 folds: ')
print('   ')
print('Average accuracy over 10 folds: ' + str(total_accuracy))



