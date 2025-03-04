import pandas as pd
import numpy as np
import re
import joblib
import codecarbon
from datasets import load_dataset,Dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from codecarbon import EmissionsTracker
import lightgbm as lgb
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
import evaluate

# 1. Chargement des données
def load_data():
    dataset = load_dataset("QuotaClimat/frugalaichallenge-text-train")
    df = pd.DataFrame(dataset['train'])
    return df

# 2. Prétraitement des textes
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

def preprocess_data(df):
    df['quote'] = df['quote'].apply(preprocess_text)
    return df

# 3. Vectorisation des textes
def vectorize_texts(train_texts, test_texts):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer

# 4. Entraînement de plusieurs modèles avec suivi de l'empreinte carbone
def train_model(X_train, y_train, model_type="logistic_regression"):
    tracker = EmissionsTracker()
    tracker.start()
    
    if model_type == "logistic_regression":
        model = LogisticRegression(max_iter=500)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=50, max_depth=10)
    elif model_type == "lightgbm":
        model = lgb.LGBMClassifier(n_estimators=50, num_leaves=31)
    else:
        raise ValueError("Modèle non reconnu")
    
    model.fit(X_train, y_train)
    emissions = tracker.stop()
    
    print(f"Consommation carbone pour {model_type}: {emissions:} kgCO2eq")
    return model

def train_transformer_model(model_name, train_texts, train_labels,test_texts,test_labels):
    tracker = EmissionsTracker()
    tracker.start()

    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)  # Liste d'entiers
    test_labels_encoded = label_encoder.transform(test_labels)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator=DataCollatorWithPadding(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=8)
    #model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    
    train_text_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=128)
    train_dataset = Dataset.from_dict({'input_ids': train_text_encodings['input_ids'], 'labels': train_labels_encoded})

    test_text_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, max_length=128)
    test_dataset = Dataset.from_dict({'input_ids': test_text_encodings['input_ids'], 'labels': test_labels_encoded})

    training_args = TrainingArguments(
        "test-trainer",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=8,
        learning_rate=2e-5,
        weight_decay=0.01,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    trainer.train()
    emissions:float = tracker.stop()
    
    print(emissions)

    predictions, label_ids, metrics = trainer.predict(test_dataset)
    predicted_labels = predictions.argmax(axis=-1)

    decoded_predictions = label_encoder.inverse_transform(predicted_labels)
    decoded_labels = label_encoder.inverse_transform(test_labels_encoded)

    for i in range(5):
        print(f"Texte: {test_texts.iloc[i]}")
        print(f"Prédiction: {decoded_predictions[i]}")
        print(f"Étiquette réelle: {decoded_labels[i]}")
        print()

    metric = evaluate.load("accuracy")
    metric.compute(predictions=predicted_labels, references=label_ids) 

    return model, tokenizer,metrics,metric




# 6. Évaluation du modèle
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

# 7. Pipeline principal
def main():
    df = load_data()
    df = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(df['quote'], df['label'], test_size=0.2, random_state=42)
    X_train_vec, X_test_vec, vectorizer = vectorize_texts(X_train, X_test)
    
    models = ["logistic_regression", "random_forest", "lightgbm"]
    for model_type in models:
        print(f"--- Entraînement de {model_type} ---")
        model = train_model(X_train_vec, y_train, model_type)
        evaluate_model(model, X_test_vec, y_test)
        joblib.dump(model, f'{model_type}_model.pkl')
    
    # Entraînement de modèles transformers légers
    transformer_models = ["distilbert-base-uncased"]
    for t_model in transformer_models:
        print(f"--- Entraînement de {t_model} ---")
        model, tokenizer,metrics,metric = train_transformer_model(t_model, X_train, y_train,X_test,y_test)
    
if __name__ == '__main__':
    main()
