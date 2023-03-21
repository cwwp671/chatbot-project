import unittest
import json
import os
from src.chatbot.utils import load_json_file, save_json_file, create_directory_if_not_exists, preprocess_and_save_data


class TestUtils(unittest.TestCase):

    def setUp(self):
        # Set up variables used in the tests
        self.raw_data_file = 'data/raw_data/conversations.json'
        self.processed_data_file = 'data/processed_data/tokenized_data.json'
        self.test_directory = 'data/test_directory'

    def test_load_json_file(self):
        # Test load_json_file function
        data = load_json_file(self.raw_data_file)

        # Assert the data is not None and is a dictionary
        self.assertIsNotNone(data)
        self.assertIsInstance(data, dict)

    def test_save_json_file(self):
        # Test save_json_file function
        test_file = 'data/test.json'
        test_data = {'test': 'data'}

        save_json_file(test_file, test_data)

        # Assert the saved data matches the expected data
        with open(test_file, 'r') as f:
            loaded_data = json.load(f)

        self.assertEqual(test_data, loaded_data)

        # Remove the test file
        os.remove(test_file)

    def test_create_directory_if_not_exists(self):
        # Test create_directory_if_not_exists function
        create_directory_if_not_exists(self.test_directory)

        # Assert the directory has been created
        self.assertTrue(os.path.exists(self.test_directory))

        # Remove the test directory
        os.rmdir(self.test_directory)

    def test_preprocess_and_save_data(self):
        # Test preprocess_and_save_data function
        preprocess_and_save_data(self.raw_data_file, self.processed_data_file)

        # Assert the processed data file has been created
        self.assertTrue(os.path.exists(self.processed_data_file))


if __name__ == '__main__':
    unittest.main()
