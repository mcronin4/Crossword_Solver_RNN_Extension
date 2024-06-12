import pandas as pd
from transformers import RobertaTokenizer, TFRobertaModel
import tensorflow as tf
from tensorflow.python.keras.utils import pad_sequences



data = pd.read_csv('nytcrosswords.csv', usecols=['Word', 'Clue'])

print(data.head())

clues = data['clue'].astype(str).values
answers = data['answer'].astype(str).values

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
input_ids = [tokenizer.encode(clue, add_special_tokens=True) for clue in clues]

max_length = max([len(ids) for ids in input_ids])
input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long", truncating="post", padding="post")
input_ids = tf.convert_to_tensor(input_ids)

answer_tokenizer = tf.python.keras.preprocessing.text.Tokenizer(char_level=True)
answer_tokenizer.fit_on_texts(answers)
answer_sequences = answer_tokenizer.texts_to_sequences(answers)



