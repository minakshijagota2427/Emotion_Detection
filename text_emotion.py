import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------------- BIGGER DATASET ----------------
texts = [
    # HAPPY
    "I am very happy",
    "I feel amazing",
    "This is wonderful",
    "I am excited",
    "I love this",
    "Feeling great today",
    "Life is beautiful",
    "I am so joyful",
    "Everything is awesome",
    "I am smiling",
    
    # SAD
    "I am very sad",
    "I feel depressed",
    "This is terrible",
    "I am unhappy",
    "I feel lonely",
    "I want to cry",
    "Feeling down today",
    "I am broken",
    "Life is painful",
    "I feel hopeless",
    
    # ANGRY
    "I am angry",
    "I am furious",
    "This makes me mad",
    "I am frustrated",
    "I hate this",
    "This is annoying",
    "I am irritated",
    "I feel rage",
    "So much anger inside me",
    "I am losing my temper",
    
    # FEAR
    "I am scared",
    "I feel afraid",
    "This is frightening",
    "I am terrified",
    "I am nervous",
    "I feel unsafe",
    "I am worried",
    "I feel panic",
    "This is dangerous",
    "I am shaking with fear",
    
    # SURPRISE
    "I am surprised",
    "Oh wow",
    "This is shocking",
    "I did not expect this",
    "Unbelievable",
    "That is amazing",
    "What a surprise",
    "I am shocked",
    "This is unexpected",
    "Wow this is crazy",
    
    # NEUTRAL
    "I am okay",
    "I feel normal",
    "Nothing special",
    "Just another day",
    "I am fine",
    "Everything is normal",
    "I feel balanced",
    "No strong feelings",
    "Just working",
    "I am calm"
]

labels = (
    ["Happy"]*10 +
    ["Sad"]*10 +
    ["Angry"]*10 +
    ["Fear"]*10 +
    ["Surprise"]*10 +
    ["Neutral"]*10
)

# ---------------- MODEL ----------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

# ---------------- SAVE ----------------
pickle.dump(model, open("text_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Text model trained & saved!")