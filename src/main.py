from src.chatbot.model import MyChatbotModel
from src.chatbot.utils import preprocess_and_save_data


def main():
    # Set file paths for raw data, processed data, and trained model
    raw_data_file = 'data/raw_data/conversations.json'
    processed_data_file = 'data/processed_data/tokenized_data.json'
    trained_model_file = 'data/model/trained_model.pickle'

    # Preprocess the raw data and save it as a processed data file
    preprocess_and_save_data(raw_data_file, processed_data_file)

    # Initialize and train the chatbot model
    chatbot_model = MyChatbotModel()
    chatbot_model.train(processed_data_file)

    # Save the trained model as a pickle file
    chatbot_model.save(trained_model_file)

    # Load the trained model from the pickle file
    chatbot_model.load(trained_model_file)

    # Use the chatbot model to generate responses
    while True:
        # Get user input
        user_input = input("You: ")

        # If the user wants to quit, exit the chatbot
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Chatbot: Goodbye!")
            break
        # Otherwise, generate a response from the chatbot model
        else:
            response = chatbot_model.generate_response(user_input)
            print("Chatbot:", response)


if __name__ == "__main__":
    main()
