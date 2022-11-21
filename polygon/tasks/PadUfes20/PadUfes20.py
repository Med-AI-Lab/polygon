import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import sklearn.metrics
from torch.utils.data import Dataset
from polygon.data.PadUfes20.PadUfes20 import PadUfes20Data
from polygon.tasks.phase import Phase


class ImageClassificationDataset(Dataset):
    def __init__(self, df, tfm=None):
        super().__init__()
        self.df = df
        self.tfm = tfm

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row.image_path).convert('RGB')
        if self.tfm is not None:
            img = self.tfm(img)
        return {'image': img, 'ID': row.ID, 'label': row.label}

def _get_diagnoses_names():
    return ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']

def _get_data(data:PadUfes20Data):
    df = data.get_metadata()
    _diagnosis_2_idx = {d: i for (i,d) in enumerate(_get_diagnoses_names())}
    df['label'] = df.diagnostic.map(lambda x: _diagnosis_2_idx[x])
    df['ID'] = df.patient_id +df.lesion_id.map(lambda x: f'___{x}')
    return df[['ID', 'image_path', 'label']]

def _make_unique_IDs(df):
    df = df.copy()
    df.ID = df.ID + df.index.map(lambda x: '_' + str(x))
    return df

def get_split_data(data:PadUfes20Data, seed:int):
    df = _get_data(data)

    IDs = sorted(set(df.ID))

    lbl_by_ID = {}
    for ID in IDs:
        df2 = df[df.ID == ID]
        assert len(set(df2.label.values)) == 1, f"labeling mismatch at ID={ID}"
        lbl_by_ID[ID] = df2.label.iloc[0]

    hold_out_size = int(len(IDs) * 0.2)
    other_IDs, test_IDs = train_test_split(IDs, test_size=hold_out_size, stratify=[lbl_by_ID[ID] for ID in IDs], random_state=seed)
    trn_IDs, val_IDs = train_test_split(other_IDs, test_size=hold_out_size, stratify=[lbl_by_ID[ID] for ID in other_IDs], random_state=seed)

    trn = df[df.ID.isin(trn_IDs)]
    val = df[df.ID.isin(val_IDs)]
    test = df[df.ID.isin(test_IDs)]

    return {
        Phase.Train: _make_unique_IDs(trn), 
        Phase.Valid: _make_unique_IDs(val), 
        Phase.Test: _make_unique_IDs(test)
    }

class PadUfes20_ImageClassification_Task:
    def __init__(self, data:PadUfes20Data, seed:int=0):
        self._data = get_split_data(data, seed)

    def get_num_classes(self): return 6

    def get_dataset(self, phase:Phase, tfm=None) -> Dataset:
        return ImageClassificationDataset(self._data[phase], tfm)

    def evaluate(self, phase, predictions):
        df = self._data[phase]
        IDs = df.ID.values.tolist()
        gt = df.label.values
        pred = np.array([predictions[ID] for ID in IDs])
        return {'balanced_accuracy': sklearn.metrics.balanced_accuracy_score(gt, pred)}
