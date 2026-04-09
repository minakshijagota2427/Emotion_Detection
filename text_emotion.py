from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Improved dataset
texts = [
    "I am very happy", "I feel great", "This is awesome", "I love this",
    "I am sad", "I feel bad", "This is terrible", "I am depressed",
    "I am angry", "I hate this", "This is frustrating",
    "Wow amazing", "This is surprising",
    "I feel okay", "Nothing special", "I am normal"
]

labels = [
    "Happy","Happy","Happy","Happy",
    "Sad","Sad","Sad","Sad",
    "Angry","Angry","Angry",
    "Surprise","Surprise",
    "Neutral","Neutral","Neutral"
]

# Better vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

# Input
user_input = input("Enter text: ")

X_test = vectorizer.transform([user_input])
prediction = model.predict(X_test)

print("Detected Emotion:", prediction[0])