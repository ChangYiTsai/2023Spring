import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer


class BERTDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.data = {}
        for idx, row in df.iterrows():
            self.data[idx] = (row['review'], row['sentiment'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        review, sentiment = self.data[idx]
        return review, torch.tensor(sentiment)


class BERT_IMDB(nn.Module):
    '''
    Fine-tuning DistillBert with two MLPs.
    '''

    def __init__(self, pretrained_type):
        super().__init__()

        num_labels = 2
        self.pretrained_model = AutoModel.from_pretrained(
            pretrained_type, num_labels=num_labels)
        
        # TO-DO 2-1: Construct a classifier
        # BEGIN YOUR CODE
        hidden_size = 768
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            #nn.Dropout(0.1),
            #nn.ReLU(),
            #nn.Linear(hidden_size // 2, hidden_size // 4),
            #nn.Dropout(0.2),
            #nn.Sigmoid(),
            nn.Linear(hidden_size // 2, 2)  # Assuming binary classification (positive/negative)
        )

        # END YOUR CODE

    def forward(self, **pretrained_text):
        outputs = self.pretrained_model(**pretrained_text).last_hidden_state
        pretrained_output = outputs[:, 0, :]
        logits = self.classifier(pretrained_output)

        return logits


class BERT():
    def __init__(self, pretrained_type, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_type)
        self.model = BERT_IMDB(pretrained_type).to(config['device'])
    
    def forward(self, text):    
        outputs = self.model(**text)
        return outputs

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()