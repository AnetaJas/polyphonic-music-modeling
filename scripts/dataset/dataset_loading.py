import torch
from torch.utils.data import Dataset
import os 
from collections import Counter


DATA_DIR = './data'
SEQ_LENGTH = 64

class MusicDataset(Dataset):
    '''
    Dataset loader for polyphonic music modeling. 
    '''
    def __init__(self, file):
        self.notes = self.load_notes(file)
        self.uniq_chars = self.get_uniq_char()
        self.char_to_index = {char: index for index, char in enumerate(self.uniq_chars)} #decoding
        self.index_to_char = {index: char for index, char in enumerate(self.uniq_chars)} #encoding
        self.char_indexes = [self.char_to_index[w] for w in self.notes]
        
    def load_notes(self, file):
        '''
        tutaj mozna cala logike zwiazana z ladowaniem danych dodac. 
        Wg mnie do zastanowienia:
            1. Dane - czy poczatek jest wazny?
            2. Batch - czy myslimy jakos w ten sposob by uczyc go jednorazowo na batchu dowolnej dlugosci, czy bardziej ze dla jednego X: dajemy uczenie i potem dla kolejnego itd...'
        Na razie opcja brute force - wczytuje plik i jade z nim po kolei. 
        '''
        text = open(os.path.join(DATA_DIR, file)).read()
        return text
    
    def get_uniq_char(self):
        char_count = Counter(self.words)
        return sorted(char_count, key=char_count.get, reverse=True)
    
    def __len__(self):
        return len(self.words_indexes) - SEQ_LENGTH
    
    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index+SEQ_LENGTH]), # X
            torch.tensor(self.words_indexes[index+1:index+SEQ_LENGTH+1]), # Y
        )
        