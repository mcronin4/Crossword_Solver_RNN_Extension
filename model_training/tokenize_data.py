import pandas as pd
import numpy as np
from transformers import RobertaTokenizer, TFRobertaModel
import tensorflow as tf
import tf_keras
from tf_keras.layers import Input, LSTM, Dense
from tf_keras.models import Model
from sklearn.model_selection import train_test_split
from keras import preprocessing


class ClueModel:
    def __init__(self, model_name='roberta-base'):
        self.model_name = model_name
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.char_to_index = {}
        self.max_length = None
        self.max_answer_length = None
        self.num_classes = None
        self.model = None
    
    def load_data(self, csv_path):
        data = pd.read_csv(csv_path, usecols=['Word', 'Clue'])
        self.clues = data['Clue'].astype(str).values
        self.answers = data['Word'].astype(str).values

    def tokenize_clues(self):
        input_ids = [self.tokenizer.encode(clue) for clue in self.clues]
        self.max_length = max(len(ids) for ids in input_ids)
        input_ids = preprocessing.sequence.pad_sequences(input_ids, maxlen=self.max_length, dtype="long", truncating="post", padding="post")
        self.input_ids = tf.convert_to_tensor(input_ids)

    def char_tokenize_answers(self):
        char_set = set(''.join(self.answers))
        self.char_to_index = {char: idx + 1 for idx, char in enumerate(char_set)}
        self.char_to_index['<PAD>'] = 0

        def char_tokenize(texts, char_to_index):
            return [[char_to_index[char] for char in text] for text in texts]

        answer_sequences = char_tokenize(self.answers, self.char_to_index)
        self.max_answer_length = max(len(seq) for seq in answer_sequences)
        answer_padded = preprocessing.sequence.pad_sequences(answer_sequences, maxlen=self.max_answer_length, dtype="long", truncating="post", padding="post")

        self.num_classes = len(self.char_to_index)
        answer_padded = np.array([tf_keras.utils.to_categorical(seq, num_classes=self.num_classes) for seq in answer_padded])
        self.answer_padded = tf.convert_to_tensor(answer_padded)
    
    def split_data(self, test_size=0.1):
        self.input_train, self.input_val, self.answer_train, self.answer_val = train_test_split(
            self.input_ids, self.answer_padded, test_size=test_size)

    def build_model(self):
        roberta_model = TFRobertaModel.from_pretrained(self.model_name)

        input_ids = Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        roberta_output = roberta_model(input_ids)[0]

        x = LSTM(128, return_sequences=True)(roberta_output)
        output = tf_keras.layers.TimeDistributed(Dense(self.num_classes, activation='softmax'))(x)

        self.model = Model(inputs=input_ids, outputs=output)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, epochs=10, batch_size=16):
        self.model.fit(
            self.input_train, self.answer_train,
            validation_data=(self.input_val, self.answer_val),
            epochs=epochs, batch_size=batch_size
        )

    def save_model(self, path):
        self.model.save(path)

clue_model = ClueModel()
clue_model.load_data('model_training\\nytcrosswords.csv')
clue_model.tokenize_clues()
clue_model.char_tokenize_answers()
clue_model.split_data()
clue_model.build_model()
clue_model.train_model()
clue_model.save_model('CWModel1.0.h5')        



