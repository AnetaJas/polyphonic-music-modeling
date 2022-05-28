import io
import os
import pathlib as p
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_DIR = "./../../../data/dataset"  # for training #TODO: poprawic tak zeby dzialalo zawsze. Niewazne jak odpalamy.
SEQ_LENGTH = 64


class MusicDataset(Dataset):
    """
    Dataset loader for polyphonic music modeling.
    """

    def __init__(self, dataset_root: str):
        self.root = dataset_root
        self.notes = self.load_notes()
        self.uniq_chars = self.get_uniq_char()
        self.char_to_index = {
            char: index for index, char in enumerate(self.uniq_chars)
        }  # decoding
        self.index_to_char = {
            index: char for index, char in enumerate(self.uniq_chars)
        }  # encoding
        self.char_indexes = []
        for note in self.notes:
            self.char_indexes.append([self.char_to_index[w] for w in note])
        # self.char_indexes =  [self.char_to_index[w] for note in self.notes for w in note]
        # self.char_indexes = [self.char_to_index[w] for w in [for note in self.notes]]

    def load_notes(self):
        """
        tutaj mozna cala logike zwiazana z ladowaniem danych dodac.
        Wg mnie do zastanowienia:
            1. Dane - czy poczatek jest wazny?
            2. Batch - czy myslimy jakos w ten sposob by uczyc go jednorazowo na batchu dowolnej dlugosci, czy bardziej ze dla jednego X: dajemy uczenie i potem dla kolejnego itd...'
        Na razie opcja brute force - wczytuje plik i jade z nim po kolei.
        """
        self.text = "".join(
            [
                open(os.path.join(self.root, "preprocessed", file)).read()
                for file in os.listdir(p.Path(self.root, "preprocessed"))
            ]
        )

        self.text_array = []

        text = io.StringIO(self.text)
        lines = text.readlines()

        index = 0
        max_index = len(lines)
        connected_track = ""

        while index < max_index:
            track = []
            if lines[index] == "\n":
                index += 1
            else:
                while lines[index] != "\n":
                    track.append(lines[index][:-1])
                    index += 1
                connected_track = "".join(track)
                self.text_array.append(connected_track)

        return self.text_array

    def get_uniq_char(self):
        char_count = Counter(self.text)
        return sorted(char_count, key=char_count.get, reverse=True)

    def __len__(self):
        return len(self.char_indexes)

    def __getitem__(self, index):
        return (
            torch.tensor((self.char_indexes[index])[:-1]),  # X
            torch.tensor((self.char_indexes[index])[1:]),  # Y
        )

    def get_encoded_item(self, index):
        return [
            (self.notes[index])[:-1],  # X
            (self.notes[index])[1:],
        ]  # Y

    def ret_dimension(self):
        return len(self.uniq_chars)
