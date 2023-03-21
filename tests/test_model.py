import unittest
from src.chatbot.model import MyChatbotModel
from src.chatbot.utils import preprocess_and_save_data


class TestMyChatbotModel(unittest.TestCase):

    def setUp(self):
        self.raw_data_file = 'data/raw_data/conversations.json'
        self.processed_data_file = 'data/processed_data/tokenized_data.json'
        self.trained_model_file = 'data/model/trained_model.pickle'

        preprocess_and_save_data(self.raw_data_file, self.processed_data_file)

        self.chatbot_model = MyChatbotModel()
        self.chatbot_model.train(self.processed_data_file)

    def test_generate_response(self):
        response = self.chatbot_model.generate_response("What's your name?")
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)

    def test_save_and_load_model(self):
        self.chatbot_model.save(self.trained_model_file)

        loaded_model = MyChatbotModel.load(self.trained_model_file)
        self.assertIsNotNone(loaded_model)
        self.assertIsInstance(loaded_model, MyChatbotModel)

        response = loaded_model.generate_response("What's your name?")
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)


if __name__ == '__main__':
    unittest.main()
