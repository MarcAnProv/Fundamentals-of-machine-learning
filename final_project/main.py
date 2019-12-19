import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns; sns.set_style('darkgrid')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from nltk import download

import project_util

download('wordnet')

print('Getting sentiment data...')
sentiment_data = project_util.get_data_sentiment()
print('Getting personality data...')
personality_data = project_util.get_data_personality()


fig = plt.figure(figsize=(22,8))
np.random.seed(0)

print('Logistic regression , sentiment dataset...')
subsets = range(0,150,15)
clf, X_test, y_test, vectorizer = project_util.train(data=sentiment_data, clf=LogisticRegression(C=3.5, class_weight='balanced'))
scores = project_util.predictOnSubsets(clf, X_test, y_test, vectorizer, subsets, min_words=150)

ax = plt.subplot(241)
sns.lineplot(x=subsets, y=scores[0])
#ax.set_ylim(0,100)
ax.set_xlabel('Text sample length')
ax.set_ylabel('Accuracy')
ax.title.set_text('accuracy, sentiment dataset')

ax = plt.subplot(242)
sns.lineplot(x='subset', y='confidence', data=scores[1])
#ax.set_ylim(0,100)
ax.set_xlabel('Text sample length')
ax.set_ylabel('Mean probability of predicted class')
ax.title.set_text('prediction confidence, sentiment dataset')


print('Logistic regression, personality dataset...')
subsets = range(0,500,50)
clf, X_test, y_test, vectorizer = project_util.train(data=personality_data, clf=LogisticRegression(C=0.1, class_weight='balanced'))
scores = project_util.predictOnSubsets(clf, X_test, y_test, vectorizer, subsets, min_words=1000)

ax = plt.subplot(243)
sns.lineplot(x=subsets, y=scores[0])
#ax.set_ylim(0,100)
ax.set_xlabel('Text sample length')
ax.set_ylabel('Accuracy')
ax.title.set_text('accuracy, personality dataset')

ax = plt.subplot(244)
sns.lineplot(x='subset', y='confidence', data=scores[1])
#ax.set_ylim(0,100)
ax.set_xlabel('Text sample length')
ax.set_ylabel('Mean probability of predicted class')
ax.title.set_text('prediction confidence, personality dataset')

""" 
X_sentiment = sentiment_data.drop(['sentiment'], axis=1)
y_sentiment = sentiment_data['sentiment']

sentiment_X_train, sentiment_x_test, sentiment_y_train, sentiment_y_test = project_util.train_test_split(X_sentiment, y_sentiment, test_size=0.2)

vectorizer = project_util.TfidfVectorizer(stop_words='english')
sentiment_X_train = vectorizer.fit_transform(sentiment_X_train['review'])
sentiment_x_test = vectorizer.transform(sentiment_x_test['review'])

print('hyperparameter search, sentiment dataset')
grid_values = {'penalty': ['l2'],'C':[0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5]}
grid_clf = GridSearchCV(LogisticRegression(solver='sag', class_weight='balanced', multi_class='ovr'), param_grid = grid_values, scoring = 'accuracy', cv=5)

best_model = grid_clf.fit(sentiment_X_train, sentiment_y_train)
print('Best C:', best_model.best_estimator_.get_params()['C'])

c_values = [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5]

project_util.plot_grid_search(grid_clf.cv_results_, c_values, 'Regularization hyperparameter value') """




""" train_sentiment = []
test_sentiment = [] """


""" for i in c_values:
    clf = LogisticRegression(C=i, solver='sag', class_weight='balanced', multi_class='ovr')
    clf.fit(sentiment_X_train, sentiment_y_train)
    test_sentiment.append(clf.score(sentiment_x_test, sentiment_y_test))
    train_pred = clf.predict(sentiment_X_train)
    train_sentiment.append((train_pred == sentiment_y_train).mean())

ax = plt.subplot(245)
sns.lineplot(x=c_values, y=train_sentiment)
sns.lineplot(x=c_values, y=test_sentiment)
ax.set_ylim(0,100)
ax.legend(['training set', 'test set'])
ax.set_xlabel('Regularization parameter values')
ax.set_ylabel('Accuracy')
ax.title.set_text('Hyperparameter search, sentiment dataset') """


x_personality = personality_data.drop(['type'], axis=1)
y_personality = personality_data['type']

personality_X_train, personality_x_test, personality_y_train, personality_y_test = project_util.train_test_split(x_personality, y_personality, test_size=0.2)

vectorizer = project_util.TfidfVectorizer(stop_words='english')
personality_X_train = vectorizer.fit_transform(personality_X_train['posts'])
personality_x_test = vectorizer.transform(personality_x_test['posts'])

print('hyperparameter search personality dataset')
grid_values = {'penalty': ['l2'],'C':[0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5]}
grid_clf = GridSearchCV(LogisticRegression(solver='sag', class_weight='balanced', multi_class='ovr'), param_grid = grid_values, scoring = 'accuracy', cv=5)

best_model = grid_clf.fit(personality_X_train, personality_y_train)
print('Best C:', best_model.best_estimator_.get_params()['C'])

c_values = [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5]

project_util.plot_grid_search(grid_clf.cv_results_, c_values, 'log10(Regularization hyperparameter value)')

""" train_personality = []
test_personality = [] """

""" for i in c_values:
    clf = LogisticRegression(C=i, solver='sag', class_weight='balanced', multi_class='ovr')
    clf.fit(personality_X_train, personality_y_train)
    test_personality.append(clf.score(personality_x_test, personality_y_test))
    train_pred = clf.predict(personality_X_train)
    train_personality.append((train_pred == personality_y_train).mean())

ax = plt.subplot(246)
sns.lineplot(x=c_values, y=train_personality)
sns.lineplot(x=c_values, y=test_personality)
ax.set_ylim(0,100)
ax.legend(['training set', 'test set'])
ax.set_xlabel('Regularization parameter values')
ax.set_ylabel('Accuracy')
ax.title.set_text('Hyperparameter search, personality dataset')"""

plt.show()
