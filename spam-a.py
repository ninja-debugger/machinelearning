'''
Here we will look to create a frequency matrix on a smaller document set to make sure
we understand how the
document-term matrix generation happens. We have created a sample document set 'documents'.
'''
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.naive_bayes import MultinomialNB

documents = ['Hello, how are you!',
'Win money, win from home.',
'Call me now.',
'Hello, Call hello you tomorrow?']

count_vector = CountVectorizer()

count_vector.fit(documents)
count_vector.get_feature_names()

doc_array = count_vector.transform(documents).toarray()

frequency_matrix=pd.DataFrame(doc_array, columns=count_vector.get_feature_names())


df = pd.read_table('spam/SMSSpamCollection', sep='\t', header=None, names=['label', 'sms_message'])

df['label'] = df.label.map({'ham':0, 'spam':1})

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                    df['label'],
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))


# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)


naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

predictions = naive_bayes.predict(testing_data)

print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))
