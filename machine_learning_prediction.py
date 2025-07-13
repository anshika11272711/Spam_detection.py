# Import necessary libraries
import pandas as pd
import joblib  # ✅ Modern way to save/load models
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import os

# Load the dataset
def load_dataset():
    data = {
        'text': [
            'Free entry in 2 a wkly comp to win a £1000 gift card',
            'Call me when you get this message',
            'Congratulations, you’ve won a free ticket to the Bahamas!',
            'Hey, are you coming to the party tonight?',
            'Get paid to take surveys online',
            'Happy birthday! Enjoy your special day!',
            'Limited offer: Free iPhone, act fast!',
            'Hey, I need to talk to you about the project.',
            'Special offer on your credit card, don’t miss out!',
            'I am looking forward to our dinner tomorrow.',
            'Free vacation to Hawaii – click here to claim!',
            'Let’s discuss the meeting notes when you get a chance.',
            'Congratulations, you’ve won a lottery of $1000!'
        ],
        'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam']
    }
    df = pd.DataFrame(data)
    return df

# Preprocess the data
def preprocess_data(df):
    X = df['text']
    y = df['label']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
def vectorize_data(X_train, X_test):
    vectorizer = CountVectorizer(stop_words='english')
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)
    return vectorizer, X_train_vect, X_test_vect

# Train the model
def train_model(X_train_vect, y_train):
    model = MultinomialNB()
    model.fit(X_train_vect, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test_vect, y_test):
    y_pred = model.predict(X_test_vect)
    print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
def save_model(model, vectorizer):
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/spam_classifier_model.pkl")
    joblib.dump(vectorizer, "model/spam_vectorizer.pkl")
    print("💾 Model and vectorizer saved in 'model/' folder.")

# Load the model (optional)
def load_saved_model():
    model = joblib.load("model/spam_classifier_model.pkl")
    vectorizer = joblib.load("model/spam_vectorizer.pkl")
    print("✅ Model loaded.")
    return model, vectorizer

# Main function
def main():
    df = load_dataset()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    vectorizer, X_train_vect, X_test_vect = vectorize_data(X_train, X_test)
    model = train_model(X_train_vect, y_train)
    evaluate_model(model, X_test_vect, y_test)
    save_model(model, vectorizer)
    # model, vectorizer = load_saved_model()  # Uncomment if testing loading

# Run the program
if __name__ == "__main__":
    main()
