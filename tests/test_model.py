import unittest
from src.chatbot.model import MyChatbotModel
from src.chatbot.utils import preprocess_and_save_data


class TestMyChatbotModel(unittest.TestCase):

    def setUp(self):
        # Set up the paths for the raw data file, processed data file, and trained model file
        self.raw_data_file = 'data/raw_data/conversations.json'
        self.processed_data_file = 'data/processed_data/tokenized_data.json'
        self.trained_model_file = 'data/model/trained_model.pickle'

        # Preprocess and save the data
        preprocess_and_save_data(self.raw_data_file, self.processed_data_file)

        # Initialize and train the chatbot model
        self.chatbot_model = MyChatbotModel()
        self.chatbot_model.train(self.processed_data_file)

    def test_generate_response(self):
        # Test the generate_response method of the chatbot model
        response = self.chatbot_model.generate_response("What's your name?")
        # Check if the response is not None and is of string type
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)

    def test_save_and_load_model(self):
        # Test the save and load methods of the chatbot model
        # Save the trained model as a pickle file
        self.chatbot_model.save(self.trained_model_file)

        # Load the trained model from the pickle file
        loaded_model = MyChatbotModel.load(self.trained_model_file)

        # Check if the loaded model is not None and is of MyChatbotModel type
        self.assertIsNotNone(loaded_model)
        self.assertIsInstance(loaded_model, MyChatbotModel)

        # Check if the loaded model can generate a response
        response = loaded_model.generate_response("What's your name?")
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)


if __name__ == '__main__':
    unittest.main()
