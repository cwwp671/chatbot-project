import unittest
from src.chatbot.preprocess import tokenize, preprocess_conversations, preprocess_raw_data
from src.chatbot.utils import load_json_file, create_directory_if_not_exists


class TestPreprocess(unittest.TestCase):

    def setUp(self):
        self.raw_data_file = 'data/raw_data/conversations.json'
        self.processed_data_file = 'data/processed_data/tokenized_data.json'

        create_directory_if_not_exists('data/processed_data')
        preprocess_raw_data(self.raw_data_file, self.processed_data_file)

    def test_tokenize(self):
        text = "Hello, how are you?"
        tokens = tokenize(text)

        print(tokens)  # Add this line to print the tokens

        self.assertIsNotNone(tokens)
        self.assertIsInstance(tokens, list)
        self.assertEqual(len(tokens), 4)  # Update the expected number of tokens here

    def test_preprocess_conversations(self):
        tokenized_conversations = load_json_file(self.processed_data_file)['tokenized_conversations']
        all_sentences, all_responses = preprocess_conversations(tokenized_conversations)

        self.assertIsNotNone(all_sentences)
        self.assertIsInstance(all_sentences, list)
        self.assertGreater(len(all_sentences), 0)

        self.assertIsNotNone(all_responses)
        self.assertIsInstance(all_responses, list)
        self.assertGreater(len(all_responses), 0)


if __name__ == '__main__':
    unittest.main()
