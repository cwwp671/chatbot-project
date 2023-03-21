import json
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.chatbot.preprocess import tokenize, preprocess_conversations


class MyChatbotModel:
    def __init__(self):
        # Initialize the TfidfVectorizer object
        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=True)
        # Initialize the lists that will hold all sentences and responses
        self.all_sentences = []
        self.all_responses = []
        # Initialize the matrix that will hold the tf-idf values for all sentences
        self.tfidf_matrix = None

    def train(self, processed_data_file):
        # Load the preprocessed conversations from the given file
        with open(processed_data_file, 'r') as f:
            processed_data = json.load(f)
            tokenized_conversations = processed_data['tokenized_conversations']

        # Preprocess the conversations to get all sentences and responses
        self.all_sentences, self.all_responses = preprocess_conversations(tokenized_conversations)
        # Compute the tf-idf matrix for all sentences using the TfidfVectorizer
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.all_sentences)

    def generate_response(self, user_input):
        # Compute the tf-idf vector for the user's input using the TfidfVectorizer
        user_input_vector = self.tfidf_vectorizer.transform([user_input])
        # Compute the cosine similarity between the user's input vector and all sentence vectors
        similarity_scores = cosine_similarity(user_input_vector, self.tfidf_matrix)
        # Find the index of the most similar sentence
        most_similar_index = np.argmax(similarity_scores)
        # Get the similarity score of the most similar sentence
        most_similar_score = similarity_scores[0][most_similar_index]

        # If the most similar score is 0, the model did not find a match
        if most_similar_score == 0:
            return "I'm sorry, I don't understand your question. Can you please rephrase it?"
        # Otherwise, return the response corresponding to the most similar sentence
        else:
            return self.all_responses[most_similar_index]

    def save(self, model_file):
        # Save the model object to the given file using pickle
        with open(model_file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, model_file):
        # Load the model object from the given file using pickle
        with open(model_file, 'rb') as f:
            return pickle.load(f)
