import os
import json
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    """
    Custom Dataset for loading the data
    """
    
    def __init__(self, raw_data, label_dict, tokenizer, model_name):
        """
        raw_data : list of dict
            The raw data to be loaded
        label_dict : dict
            Dictionary that maps labels to integers
        tokenizer : object
            Tokenizer object to tokenize the data
        model_name : str
            Name of the model
        """
        # List of labels for classification or empty list
        label_list = list(label_dict.keys()) 
        # Different separator tokens for different models
        sep_token = ['[SEP]'] if model_name == 'bert' else ['</s>']
        dataset = list()
        # Iterate through raw data and append to the dataset
        for data in raw_data:
            tokens = data['text'].lower().split(' ')
            label_id = label_dict[data['label']]
            dataset.append((label_list + sep_token + tokens, label_id))
        self._dataset = dataset

    def __getitem__(self, index):
        """
        index : int
            Index of the item to be fetched
        """
        return self._dataset[index]

    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self._dataset)

def my_collate(batch):
    """
    Function to collate the data
    """
    input_ids = []
    attention_masks = []
    label_ids = []
    
    for tokenized_input, label_id in batch:
        input_ids.append(tokenized_input['input_ids'])
        attention_masks.append(tokenized_input['attention_mask'])
        label_ids.append(label_id)

    # Convert to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    label_ids = torch.tensor(label_ids)

    return input_ids, attention_masks, label_ids


def load_data(data, tokenizer, batch_size, workers):
    """
    Function to load the SNLI data
    """
    # Define the label dictionary for SNLI dataset
    label_dict = {"entailment": 0, "contradiction": 1, "neutral": 2}

    # Initialize the custom dataset
    dataset = MyDataset(data, label_dict, tokenizer)

    # Initialize the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=my_collate)

    return dataloader