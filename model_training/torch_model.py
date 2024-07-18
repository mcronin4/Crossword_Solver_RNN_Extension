import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer, RobertaModel

class CrosswordDataset(Dataset):
    def __init__(self, clues, answers, tokenizer, char_to_index, max_length, max_answer_length, num_classes):
        self.clues = clues
        self.answers = answers
        self.tokenizer = tokenizer
        self.char_to_index = char_to_index
        self.max_length = max_length
        self.max_answer_length = max_answer_length
        self.num_classes = num_classes

    def __len__(self):
        return len(self.clues)

    def __getitem__(self, idx):
        clue = self.clues[idx]
        answer = self.answers[idx]
        
        input_ids = self.tokenizer.encode(clue, max_length=self.max_length, padding='max_length', truncation=True)
        
        answer_sequence = [self.char_to_index[char] for char in answer]
        answer_sequence = answer_sequence + [self.char_to_index['<PAD>']] * (self.max_answer_length - len(answer_sequence))
        answer_one_hot = np.eye(self.num_classes)[answer_sequence]
        
        return torch.tensor(input_ids), torch.tensor(answer_one_hot, dtype=torch.float32)

class ClueModel(nn.Module):
    def __init__(self, model_name, num_classes, max_length, max_answer_length):
        super(ClueModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.lstm = nn.LSTM(input_size=self.roberta.config.hidden_size, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, input_ids):
        attention_mask = (input_ids != 1).int()
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        lstm_out, _ = self.lstm(sequence_output)
        lstm_out = lstm_out.contiguous().view(-1, lstm_out.shape[-1])
        logits = self.fc(lstm_out)
        return logits.view(-1, self.max_answer_length, self.num_classes)

def load_data(csv_path):
    data = pd.read_csv(csv_path)
    clues = data['clue'].astype(str).values
    answers = data['answer'].astype(str).values
    return clues, answers

def char_tokenize_answers(answers):
    char_set = set(''.join(answers))
    char_to_index = {char: idx + 1 for idx, char in enumerate(char_set)}
    char_to_index['<PAD>'] = 0
    num_classes = len(char_to_index)
    return char_to_index, num_classes

def main():
    model_name = 'roberta-base'
    csv_path = 'model_training/nytcrosswords.csv'
    max_length = 114
    max_answer_length = 22

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    clues, answers = load_data(csv_path)
    char_to_index, num_classes = char_tokenize_answers(answers)

    dataset = CrosswordDataset(clues, answers, tokenizer, char_to_index, max_length, max_answer_length, num_classes)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ClueModel(model_name, num_classes, max_length, max_answer_length).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        for input_ids, answers in train_loader:
            input_ids = input_ids.to(device)
            answers = answers.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            
            loss = criterion(outputs.view(-1, num_classes), answers.view(-1, num_classes))
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for input_ids, answers in val_loader:
                input_ids = input_ids.to(device)
                answers = answers.to(device)
                
                outputs = model(input_ids)
                loss = criterion(outputs.view(-1, num_classes), answers.view(-1, num_classes))
                total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Validation Loss: {total_loss / len(val_loader)}")

    torch.save(model.state_dict(), 'clue_model.pth')

if __name__ == "__main__":
    main()
    



