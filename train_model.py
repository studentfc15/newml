import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from utils.db import get_data_from_supabase

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def train_and_save_model():
    df = get_data_from_supabase()
    df['cleaned_ingredients'] = df['ingredients'].apply(clean_text)

    # TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['cleaned_ingredients'])

    # Label encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['food_id'])
    y_cat = to_categorical(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y_cat, test_size=0.2, random_state=42)

    # Model
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(y_cat.shape[1], activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

    # Save model dan encoder
    model.save("model/model.h5")
    joblib.dump(vectorizer, "model/vectorizer.pkl")
    joblib.dump(label_encoder, "model/label_encoder.pkl")

    return "Model retrained & saved successfully!"
