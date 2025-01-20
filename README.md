# AI-Chatbot-with-NLP
# AI Chatbot using DialoGPT

This project implements a simple AI chatbot using the `DialoGPT` model from the Hugging Face library. The chatbot is built with a graphical user interface (GUI) using the `tkinter` library. The model generates conversational responses based on user input, with custom greetings and goodbye responses as well.

## Features

- **Conversational AI**: The chatbot uses the `microsoft/DialoGPT-medium` model to generate responses.
- **Custom Greetings and Goodbye**: The chatbot can respond to user greetings and goodbyes with predefined responses.
- **User Interface**: The chatbot is equipped with a simple and clean GUI for easy interaction.
- **History Tracking**: The chatbot maintains a conversation history that is displayed in the UI.

## Prerequisites

Before running the code, you need to have the following Python libraries installed:

- `nltk` - Natural Language Toolkit for text processing.
- `tkinter` - Built-in library for creating GUI applications.
- `transformers` - For loading and using pre-trained language models like DialoGPT.

## Example Interaction
User: "Hello"

Bot: "Hi there! How can I assist you?"

User: "What's your name?"

Bot: "I'm an AI chatbot. How can I help you today?"

User: "Thank you"

Bot: "You're welcome!"

You can install the necessary dependencies using `pip`:

```bash
pip install nltk transformers
