from keras.models import load_model
from transformers import RobertaTokenizer, TFRobertaModel
import tensorflow as tf
import numpy as np
from transformers import RobertaTokenizer
import pandas as pd


model = load_model('CWModel1.0.h5', custom_objects={"TFRobertaModel":TFRobertaModel})

data = pd.read_csv('model_training/nytcrosswords.csv', usecols=['Word', 'Clue'])
max_answer_length = 22

answers = data['Word'].astype(str).values
clues = data['Clue'].astype(str).values



# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Example input sentences
input_sentences = ["hello", "Mischief-makers", "The best person in the world", "What wiggly lines in comics may represent"]

# Tokenize the input
tokenized_inputs = [tokenizer.encode(input, max_length=max_answer_length, truncation=True, padding='max_length') for input in input_sentences]
max_length = max_answer_length  # Ensuring max_length matches max_answer_length
tokenized_inputs = tf.convert_to_tensor(tokenized_inputs)

# Convert the BatchEncoding to a dictionary of tensors
# inputs = {key: value for key, value in tokenized_inputs.items()}
# Assuming 'model' is already loaded
# Use the tokenized inputs for prediction
predictions = model.predict(tokenized_inputs)

print(predictions)
print(str(len(predictions))+" and "+str(len(predictions[0]))+" and "+str(len(predictions[0][0])))

char_set = set(''.join(answers))
print(char_set)
char_to_index = {char: idx + 1 for idx, char in enumerate(char_set)}
char_to_index['<PAD>'] = 0
print(len(char_to_index))
index_to_char = {v: k for k, v in char_to_index.items()}  # reverse the char_to_index dictionary


# Detokenize the predictions
def detokenize_predictions(predictions, index_to_char):
    predicted_words = []
    for sequence in predictions:
        word = ''
        for timestep in sequence:
            char_index = np.argmax(timestep)  # get the index of the max probability
            char = index_to_char.get(char_index, '')  # convert index back to character
            if char != '<PAD>':  # skip padding tokens
                word += char
        predicted_words.append(word)
        print(word)
    return predicted_words

# Get the predicted words
predicted_words = detokenize_predictions(predictions, index_to_char)

# Output the predicted words
for word in predicted_words:
    print(word)

print(char_to_index)
print(index_to_char)
