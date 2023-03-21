import json
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.chatbot.preprocess import tokenize, preprocess_conversations


class MyChatbotModel:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=True)
        self.all_sentences = []
        self.all_responses = []
        self.tfidf_matrix = None

    def train(self, processed_data_file):
        with open(processed_data_file, 'r') as f:
            processed_data = json.load(f)
            tokenized_conversations = processed_data['tokenized_conversations']

        self.all_sentences, self.all_responses = preprocess_conversations(tokenized_conversations)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.all_sentences)

    def generate_response(self, user_input):
        user_input_vector = self.tfidf_vectorizer.transform([user_input])
        similarity_scores = cosine_similarity(user_input_vector, self.tfidf_matrix)

        most_similar_index = np.argmax(similarity_scores)
        most_similar_score = similarity_scores[0][most_similar_index]

        if most_similar_score == 0:
            return "I'm sorry, I don't understand your question. Can you please rephrase it?"
        else:
            return self.all_responses[most_similar_index]

    def save(self, model_file):
        with open(model_file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, model_file):
        with open(model_file, 'rb') as f:
            return pickle.load(f)
