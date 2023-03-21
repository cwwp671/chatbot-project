import unittest
from src.chatbot.preprocess import tokenize, preprocess_conversations, preprocess_raw_data
from src.chatbot.utils import load_json_file, create_directory_if_not_exists


class TestPreprocess(unittest.TestCase):

    def setUp(self):
        self.raw_data_file = 'data/raw_data/conversations.json'
        self.processed_data_file = 'data/processed_data/tokenized_data.json'

        # Ensure that the directory for the processed data file exists, preprocess and save the data
        create_directory_if_not_exists('data/processed_data')
        preprocess_raw_data(self.raw_data_file, self.processed_data_file)

    def test_tokenize(self):
        # Test the tokenize function
        text = "Hello, how are you?"
        tokens = tokenize(text)

        self.assertIsNotNone(tokens)  # Ensure that the tokens are not None
        self.assertIsInstance(tokens, list)  # Ensure that the tokens are a list
        self.assertEqual(len(tokens), 4)  # Ensure that the number of tokens is as expected

    def test_preprocess_conversations(self):
        # Test the preprocess_conversations function
        tokenized_conversations = load_json_file(self.processed_data_file)['tokenized_conversations']
        all_sentences, all_responses = preprocess_conversations(tokenized_conversations)

        self.assertIsNotNone(all_sentences)  # Ensure that the sentences are not None
        self.assertIsInstance(all_sentences, list)  # Ensure that the sentences are a list
        self.assertGreater(len(all_sentences), 0)  # Ensure that there is at least one sentence

        self.assertIsNotNone(all_responses)  # Ensure that the responses are not None
        self.assertIsInstance(all_responses, list)  # Ensure that the responses are a list
        self.assertGreater(len(all_responses), 0)  # Ensure that there is at least one response


if __name__ == '__main__':
    unittest.main()
