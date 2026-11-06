
import pandas as pd
import numpy as np
import os
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from string import punctuation
from nltk import ngrams
import matplotlib.pyplot as plt
from __future__ import division
os.chdir("C:\\Users\\Prudhvi\\Documents\\data\\quora_qn_pairs")

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['q1_n_words'] = train['question1'].apply(lambda row: len(row.split(" ")))
train['q2_n_words'] = train['question2'].apply(lambda row: len(row.split(" ")))

train.head()

stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then','just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which','Is','If','While','This']


def text_to_wordlist(text, remove_stop_words=True, stem_words=False):
    # Clean the text, with the option to remove stop_words and to stem words.
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", str(text))
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text)
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)
    text = re.sub(r"demonitization", "demonetization", text)
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text)
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text)
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text)
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text)
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])

    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    # Optionally, shorten words to their stems
    """ 
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    """
    # Return a list of words
    return (text)

def process_questions(question_list, questions, question_list_name, dataframe):
    '''transform questions and display progress'''
    for question in questions:
        question_list.append(text_to_wordlist(question))
        if len(question_list) % 100000 == 0:
            progress = len(question_list)/len(dataframe)*100
            print("{} is {}% complete.".format(question_list_name, round(progress, 1)))


train_question1 = []
process_questions(train_question1, train.question1, 'train_question1', train)

train_question2 = [ ]
process_questions(train_question2, train.question2, 'train_question2', train)

df = pd.DataFrame()
uni_match = [ ]
bi_match = [ ]
for q1,q2 in zip(train_question1,train_question2) :
    q1_words = [word.lower() for word in q1.split()]
    q2_words = [word.lower() for word in q2.split()]
    q1_uni = list(ngrams(q1_words,1))
    q2_uni = list(ngrams(q2_words,1))
    uni_percent = len(list(set(q1_uni).intersection(q2_uni)))/np.max([len(q1_words),len(q2_words)])
    uni_match.append(uni_percent)

    q1_bi = list(ngrams(q1_words,2))
    q2_bi = list(ngrams(q2_words,2))
    bi_percent = len(list(set(q1_bi).intersection(q2_bi)))/np.max([len(q1_words),len(q2_words)])
    bi_match.append(bi_percent)

df['uni_match'] = uni_match
df['bi_match'] = bi_match
df.set_index(train.id,inplace= True)
df['target'] = train.is_duplicate
df.fillna(0,inplace=True)
""" Modelling logistic regression"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
x_train,x_test,y_train,y_test = train_test_split(df.iloc[:,:-1],df['target'])

model = LogisticRegression()
model.fit(x_train,y_train)
train_preds = model.predict(x_train)
accuracy_score(y_train,train_preds)

preds = model.predict(x_test)
accuracy_score(y_test,preds)
test_prob = model.predict_proba(x_test)
log_loss(y_test,test_prob)