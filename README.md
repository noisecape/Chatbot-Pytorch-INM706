# Conversational Chatbot in PyTorch

## Overview

This project focuses on the development of a conversational chatbot using Sequence to Sequence (Seq2Seq) models in PyTorch, employing techniques such as Long Short-Term Memory (LSTM) cells and the Attention mechanism. The chatbot is designed to interact in a human-like manner, leveraging the Cornell Movie Dialogue Corpus for training.

## Key Features

- **Seq2Seq Model**: Utilizes LSTM cells to process and generate responses based on learned dialogues from movie scripts.
- **Attention Mechanism**: Enhances the model's ability to focus on relevant parts of the input sequence, improving response accuracy.
- **Teacher Forcing Technique**: Speeds up training and helps in converging to a better solution by alternating between predicted and true next words as inputs during training.

## Experiments and Results

- **Dataset**: Cornell Movie Dialogue Corpus, featuring over 220,000 conversational exchanges.
- **Training Details**: Model trained using PyTorch with detailed logging of training and validation loss, and employed teacher forcing for efficient learning.
- **Performance Metrics**: Evaluated using BLEU scores, with detailed results indicating the effectiveness of attention mechanisms in handling long-range dependencies in dialogues.

## Usage

The chatbot is encapsulated in a user-friendly graphical interface, making it accessible for real-time interactions without the need for command-line operations. The GUI is designed with tkinter in Python.

## Conclusion

This project showcases the effectiveness of advanced NLP techniques in creating a responsive and interactive chatbot. While the BLEU score metrics suggest room for improvement, the chatbot demonstrates a solid foundation in understanding and generating human-like conversational responses.
