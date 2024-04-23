from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from keras.models import load_model
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import random
import json

app = FastAPI()

class InputData(BaseModel):
    text: str

class OutputData(BaseModel):
    response: str

# Load the model
model = load_model('chatbot_model.h5')

# Load keywords and classes_list
keywords = pickle.load(open('words.pkl', 'rb'))
classes_list = pickle.load(open('classes.pkl', 'rb'))
lemmatizer = WordNetLemmatizer()
intents_data = json.loads(open('intents.json').read())

@app.post("/predict/", response_model=OutputData)
async def predict(input_data: InputData):
    try:
        # Process input text
        tokenized_sentence = nltk.word_tokenize(input_data.text)
        preprocessed_sentence = [lemmatizer.lemmatize(word.lower()) for word in tokenized_sentence]
        bag_of_words = [0] * len(keywords)

        for token in preprocessed_sentence:
            for i, word in enumerate(keywords):
                if word == token:
                    bag_of_words[i] = 1

        # Predict intent
        predictions = model.predict(np.array([bag_of_words]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, prediction] for i, prediction in enumerate(predictions) if prediction > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        intent_list = [{"intent": classes_list[result[0]], "probability": str(result[1])} for result in results]
        predicted_intent = intent_list[0]['intent']

        # Get response based on predicted intent
        for intent_data in intents_data['intents']:
            if intent_data['tag'] == predicted_intent:
                responses = intent_data['responses']
                response = random.choice(responses)
                return {"response": response}

    except Exception as e:
        return {"response": "Error occurred: " + str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
