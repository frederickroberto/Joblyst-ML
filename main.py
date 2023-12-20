import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalMaxPooling1D, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from fastapi.responses import JSONResponse
import logging

app = FastAPI()

# Load the CSV file with job positions
df = pd.read_csv('dataset - Kualifikasi.csv')  # Replace with your actual path

# Assuming 'Job Position' is the column you are interested in
labels = df['Job Position']

# Tokenize and pad sequences
max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(labels)

x_seq = tokenizer.texts_to_sequences(labels)
x_pad = pad_sequences(x_seq, maxlen=max_len)

# Convert labels to numerical values
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Convert numerical labels to one-hot encoding
one_hot_labels = to_categorical(encoded_labels)

# Build the NLP model for classification
num_classes = len(label_encoder.classes_)
model_classification = Sequential()
model_classification.add(Embedding(input_dim=max_words, output_dim=64, input_length=max_len))
model_classification.add(Dense(512, activation='relu'))
model_classification.add(GlobalMaxPooling1D())
model_classification.add(Dropout(0.2))
model_classification.add(Dense(256, activation='relu'))
model_classification.add(Dropout(0.2))
model_classification.add(Dense(num_classes, activation='softmax'))  # Output layer for classification task

# Compile the model for classification
model_classification.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Use one-hot encoded labels as target
dummy_target_classification = np.zeros((len(x_pad), num_classes))
dummy_target_classification[:len(x_pad)] = one_hot_labels

# Train the classification model
history_classification = model_classification.fit(x_pad, dummy_target_classification, epochs=500, batch_size=64)

def get_model_predictions(input_text, threshold=0.03):
    # Tokenization and text padding
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_pad = pad_sequences(input_seq, maxlen=max_len)

    # Predict categories
    prediction = model_classification.predict(input_pad)[0]

    # Get labels that pass the threshold
    labels_passing_threshold = [label_encoder.classes_[i] for i, prob in enumerate(prediction) if prob >= threshold]

    return labels_passing_threshold

class InputText(BaseModel):
    text: str

@app.post("/predict")
async def predict(item: InputText):
    try:
        # Validate input
        if not item.text:
            raise HTTPException(status_code=400, detail="Input 'text' field cannot be empty")

        # Get predictions
        predicted_labels = get_model_predictions(item.text, threshold=0.03)

        # Add error handling to not print if no labels are found
        if not predicted_labels:
            return JSONResponse(
                content={"prediction": "Not Found"}
            )
        else:
            return JSONResponse(
                content={"prediction": predicted_labels}
            )
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Listening to http://127.0.0.1:{port}")
    uvicorn.run(app, host='127.0.0.1', port=port)
