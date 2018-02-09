import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('dataset/SMSSpamCollection.tsv', index_col=0, sep='\t')

hv = HashingVectorizer(n_features=20)
features = hv.transform(list(df["message"])).toarray()
labels = list(df["category"])

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=.1)

clf = GaussianNB()

clf.fit(features_train, labels_train)

prediction = clf.predict(features_test)

print("Test case 1:")
print("Expected label: ", labels_test[1])
print("Predicted label: ", prediction[1])

print("\nTest Case 2:")
print("Expected label: ", labels_test[165])
print("Predicted label: ", prediction[165])

print("\nAccuracy Score: ", accuracy_score(labels_test, prediction))
