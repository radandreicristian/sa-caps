import os
from pathlib import Path

import torch
import pandas as pd

from torch.utils.data import Dataset
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class NluBenchmarkDataset(Dataset):
    def __init__(self,
                 file_name: str = 'nlub.csv'):
        super(NluBenchmarkDataset, self).__init__()

        root_path = Path(__file__).parent.parent.parent.parent
        csv_path = os.path.join(root_path, 'data', 'nlu-data', file_name)

        csv = pd.read_csv(csv_path, sep=';')

        # Independent variable - Sentences containing utterances.
        x = csv.iloc[1:, 1].values

        y_slots = csv.iloc[1:, 2].values
        y_intents = csv.iloc[1:, 3].values

        n_slots = csv.iloc[1:, 2].count() + 1
        n_intents = csv.iloc[1:, 3].count()

        # Todo - Shape should be: (n_samples, max_seq_len)
        self.x = torch.tensor(x, dtype=torch.float32)

        # Todo - Shape should be: (n_samples, max_seq_len, n_slots)
        self.y_slots = torch.tensor(y_slots, )

        # Todo - Shape should be: (n_samples, n_intents)
        self.y_intents = torch.tensor(y_intents)

    def __getitem__(self, index):
        return self.x[index], self.y_slots[index], self.y_intents[index]

    def __len__(self):
        return len(self.x)
