from nltk.stem import WordNetLemmatizer
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style('darkgrid')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def get_data_sentiment(encoding=True, lengths=True):
    '''Fetches movie reviews from the 'pos' and 'neg' directories and stores them
    in a DataFrame along with their label. Returns the DataFrame.
    ---Parameters---
    encoding: boolean; default True. Whether or not to encode the labels (1/-1).
    lengths: boolean, default True. Whether or not to add a column of sample lengths.'''

    data_dict = dict({'sentiment':[], 'review':[]})
    
    for sentiment in ['pos', 'neg']:
        for filename in os.listdir(sentiment):
            file = open(sentiment+'/'+filename, encoding='utf8')
            review = file.read()
            if encoding:
                label = 1 if sentiment == 'pos' else -1
                data_dict['sentiment'].append(label)
            else:
                data_dict['sentiment'].append(sentiment)
            data_dict['review'].append(review)
            
    data = pd.DataFrame(data_dict)
    
    data['review'] = data['review'].map(process)
    
    if lengths:
        data['length'] = data['review'].map(lambda x: len(x.split(' ')))
            
    return data


def get_data_personality(filename='mbti.csv', encoding=True, lengths=True):
    '''Reads the personality type dataset and stores the posts in a DataFrame
    along with their label. Returns the DataFrame.
    ---Parameters---
    filename: string; default "mbti.csv". Name of the csv file.
    encoding: boolean; default True. Whether or not to apply label encoding.
    lengths: boolean, default True. Whether or not to add a column of sample lengths.'''
    
    data = pd.read_csv('mbti.csv')
    
    data['posts'] = data['posts'].map(lambda x: process(x.replace('|||', ' ')))
    
    data['length'] = data['posts'].map(lambda x: len(x.split(' ')))
    
    if encoding:
        le = LabelEncoder()
        data['type'] = le.fit_transform(data['type'])
        
    return data


def randomlyChooseNWords(textFile, numberOfWordsChosen):
    words = textFile.split(' ')
    randomPermutation = np.random.permutation(len(words))
    selectedWords = ""
    for i in range(numberOfWordsChosen):
        #selectedWords.append(words[randomPermutation[i]])
        selectedWords = selectedWords + " " + words[randomPermutation[i]]

    return selectedWords    


def process(text):
    tokens = text.lower().split()
    wnl = WordNetLemmatizer()
    lemmatized = [wnl.lemmatize(t) for t in tokens]
    return ' '.join(lemmatized)


def predictOnSubsets(clf, X, y, vectorizer, subsets, min_words=1000):
    accuracy = []
    confidence = {'subset':[], 'confidence':[]}
    indices = X['length'] >= min_words
    
    # Assign X and y
    try:
        X = X[indices]['posts']
    except:
        X = X[indices]['review']
    finally:
        y = y[indices]

    for i in subsets:
        X_sub = X.apply(randomlyChooseNWords, args=[i,])
        X_sub = vectorizer.transform(X_sub)
        
        # Evaluate model performance using accuracy
        accuracy.append(clf.score(X_sub, y))
        
        # Evaluate model confidence using probability of predicted class
        probabilities = np.max(clf.predict_proba(X_sub), axis=1)
        
        # Add subset identifier and probabilities to dictionary
        for p in probabilities:
            confidence['subset'].append(i)
            confidence['confidence'].append(p)
        
    return (accuracy, pd.DataFrame(confidence))


def train(data, clf):
    if 'sentiment' in data.keys():
        sample = 'review'
        label = 'sentiment'
    else:
        sample = 'posts'
        label = 'type'
        
    X = data.drop([label], axis=1)
    y = data[label]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train = vectorizer.fit_transform(X_train[sample])
    
    return clf.fit(X_train, y_train), X_test, y_test, vectorizer


def plot_grid_search(cv_results, grid_param_1, name_parameter_1,):
    #get test scores
    scores_mean = cv_results['mean_test_score']

    #plot grid search scores
    _, ax = plt.subplots(1,1)

    
    ax.plot(np.log10(grid_param_1), scores_mean)
    
    ax.set_title('Grid search scores')
    ax.set_xlabel("log10(Regularization parameter)")
    ax.set_ylabel('CV average accuracy')
    """ ax.set_xlim(range(0, 2)) """