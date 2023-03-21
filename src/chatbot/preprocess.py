import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK packages
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()


# Tokenize the text by lowercasing and removing non-alphanumeric characters
def tokenize(text):
    return [token.lower() for token in word_tokenize(text) if token.isalnum()]


# Preprocess conversations by extracting all user inputs and bot responses
def preprocess_conversations(tokenized_conversations):
    all_sentences = []
    all_responses = []

    for conversation in tokenized_conversations:
        for i, message in enumerate(conversation['messages']):
            if message['user'] == 'User':
                user_input = ' '.join(message['tokens'])
                if i + 1 < len(conversation['messages']):
                    bot_response = conversation['messages'][i + 1]['text']
                else:
                    bot_response = "I'm sorry, I don't understand your question. Can you please rephrase it?"

                all_sentences.append(user_input)
                all_responses.append(bot_response)

    return all_sentences, all_responses


# Preprocess raw data by tokenizing all messages and adding 'text' key to each message
def preprocess_raw_data(raw_data_file, processed_data_file):
    with open(raw_data_file, 'r') as f:
        raw_data = json.load(f)
        conversations = raw_data['conversations']

    tokenized_conversations = []

    for conversation in conversations:
        tokenized_messages = []
        for message in conversation['messages']:
            tokens = tokenize(message['text'])
            tokenized_messages.append({
                'user': message['user'],
                'text': message['text'],  # Include the 'text' key in the message object
                'tokens': tokens
            })
        tokenized_conversations.append({
            'id': conversation['id'],
            'messages': tokenized_messages
        })

    with open(processed_data_file, 'w') as f:
        json.dump({'tokenized_conversations': tokenized_conversations}, f)
