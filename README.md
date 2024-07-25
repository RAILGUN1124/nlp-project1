# Overview 

This repository contains respective scripts for fine tuning and evaluating a Named Entity Recognition model using respective pretrained BERT models. The model is trained on a custom dataset and uses PyTorch and the Hugging Face Transformers library.


# Description
## Data Preparation
The script reads data from CSV files, processes the data to remove punctuation, and formats it into sentences and corresponding labels. The processed data is then saved into new CSV files.

## Tokenization and Label Alignment
The script uses the BertTokenizerFast from the Hugging Face Transformers library to tokenize the input sentences. It also aligns the tokenized words with their corresponding labels to prepare the data for training.

## Dataset Class
A custom DataSequence class is defined, inheriting from torch.utils.data.Dataset. This class handles the tokenization and label alignment for each sentence and prepares batches of data for training.

## Model Definition
A custom BERT model class BertModel is defined, which uses the BertForTokenClassification from the Hugging Face Transformers library. This model is fine-tuned for the NER task.

## Training Loop
The train_loop function defines the training process:

- Initializes data loaders for the training and validation datasets.
- Sets up the optimizer and device (GPU if available).
- Iteratively trains the model over a specified number of epochs.
- Computes training and validation accuracy and loss for each epoch and prints the results.
## Evaluation
The script includes functions to evaluate a single sentence or a batch of sentences using the trained model. The predictions for the test data are generated and saved to test_ans.csv.

## Parameters
- LEARNING_RATE: Learning rate for the optimizer.
- EPOCHS: Number of epochs to train the model.
- BATCH_SIZE: Batch size for data loading.

# Requirements
To run the code, you need the following packages installed:

- pandas
- numpy
- transformers
- torch
- tqdm
