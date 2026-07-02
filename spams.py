import pandas as pd

df = pd.read_csv('spam.csv', encoding='latin-1')


df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df = df.rename(columns={'v1': 'label', 'v2': 'message'})
df.dropna(inplace=True)

print(df.head())


import re

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_message']= df['message'].apply(clean_text)
print(df.head())

df['label'] = df['label'].map({'ham':0, 'spam':1})
print(df.head())
print(df['label'].value_counts())

from sklearn.model_selection import train_test_split

x =df['clean_message']
y=df['label']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2, random_state = 42)
print(x_train.shape)
print(x_test.shape)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)
print(x_train_vectorized.shape)
print(x_test_vectorized.shape)


from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(x_train_vectorized, y_train)

y_pred = model.predict(x_test_vectorized)
print('model performance on test data')

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

print("Accuracy :", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("Precision:", round(precision_score(y_test, y_pred) * 100, 2), "%")
print("Recall   :", round(recall_score(y_test, y_pred) * 100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nDetailed Report:\n", classification_report(y_test, y_pred))