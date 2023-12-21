import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from fastapi.responses import JSONResponse
import logging
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

app = FastAPI()

# Load the CSV file with job positions
df = pd.read_csv('dataset - Kualifikasi.csv')

# Assuming 'Job Position' is the column you are interested in
labels = df['Job Position']

# Tokenize and pad sequences
max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(labels)

# Load the pre-trained model
model_classification = load_model('classification_model.h5')

# Convert labels to numerical values
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Convert numerical labels to one-hot encoding
one_hot_labels = to_categorical(encoded_labels)

def get_model_predictions(input_text, threshold=0.03):
    # Tokenization and text padding
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_pad = pad_sequences(input_seq, maxlen=max_len)

    # Predict categories using the loaded model
    prediction = model_classification.predict(input_pad)[0]

    # Get labels that pass the threshold
    labels_passing_threshold = [label_encoder.classes_[i] for i, prob in enumerate(prediction) if prob >= threshold]

    # Return "Not Found" as an array if no labels are found
    return labels_passing_threshold if labels_passing_threshold else ["Not Found"]

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

# Starting the server
# You can check the API documentation easily using /docs after the server is running
port = os.environ.get("PORT", 8080)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host='0.0.0.0',port=port)