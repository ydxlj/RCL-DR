import numpy as np
import scipy.io as sio
import torch
import scipy.sparse as sp
from torch.utils.data import Dataset
from imblearn.over_sampling import SMOTE


def load_data(filepath):
    data_dict = sio.loadmat(filepath)
    drug_disease_matrix = data_dict['didr']
    drug_names = data_dict['Wrname']
    disease_names = data_dict['Wdname']
    return drug_disease_matrix, drug_names, disease_names


class SparseMatrixDataset(Dataset):

    def __init__(self, drug_disease_matrix, apply_smote=True):
        if isinstance(drug_disease_matrix, np.ndarray):
            drug_disease_matrix = sp.csr_matrix(drug_disease_matrix)
        self.drug_disease_matrix = drug_disease_matrix
        self.n_drugs, self.n_diseases = drug_disease_matrix.shape
        self.positive_pairs = self._get_positive_pairs()
        self.negative_pairs = self._get_negative_pairs()

        print(f"Initial positive samples: {len(self.positive_pairs)}")
        print(f"Initial negative samples: {len(self.negative_pairs)}")

        if apply_smote:
            self.apply_smote()

    def _get_positive_pairs(self):
        coo = self.drug_disease_matrix.tocoo()
        return list(zip(coo.row, coo.col))

    def _get_negative_pairs(self):
        negative_pairs = []
        for i in range(self.n_drugs):
            for j in range(self.n_diseases):
                if self.drug_disease_matrix[i, j] == 0:
                    negative_pairs.append((i, j))
        return negative_pairs

    def apply_smote(self):
        X = np.vstack([self.positive_pairs, self.negative_pairs])
        y = np.hstack([
            np.ones(len(self.positive_pairs)),
            np.zeros(len(self.negative_pairs))
        ])

        smote = SMOTE(sampling_strategy=1.0, random_state=56)
        X_res, y_res = smote.fit_resample(X, y)

        self.positive_pairs = X_res[y_res == 1]
        self.negative_pairs = X_res[y_res == 0]

        print(f"After SMOTE - positive samples: {len(self.positive_pairs)}")
        print(f"After SMOTE - negative samples: {len(self.negative_pairs)}")

    def __len__(self):
        return len(self.positive_pairs)

    def __getitem__(self, idx):
        drug, pos_disease = self.positive_pairs[idx]
        _, neg_disease = self.negative_pairs[idx]
        return {
            'drug': torch.tensor(drug, dtype=torch.long),
            'pos_disease': torch.tensor(pos_disease, dtype=torch.long),
            'neg_disease': torch.tensor(neg_disease, dtype=torch.long)
        }