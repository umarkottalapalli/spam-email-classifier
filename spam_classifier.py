import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = pd.read_csv("spam.csv", encoding="latin-1")

X = data["v2"]
y = data["v1"]

y = y.map({"ham":0, "spam":1})

# clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    return text

X = X.apply(clean_text)

vectorizer = TfidfVectorizer(stop_words="english")
X_vector = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vector, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

email = input("Enter email text: ")
email = clean_text(email)

email_vec = vectorizer.transform([email])
prediction = model.predict(email_vec)

if prediction[0] == 1:
    print("Spam Email")
else:
    print("Not Spam")