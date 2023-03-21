import json
import os


def load_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def save_json_file(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def preprocess_and_save_data(raw_data_file, processed_data_file):
    from src.chatbot.preprocess import preprocess_raw_data
    create_directory_if_not_exists(os.path.dirname(processed_data_file))
    preprocess_raw_data(raw_data_file, processed_data_file)
