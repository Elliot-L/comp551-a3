import os, pickle, torch, random
import pandas as pds 
import numpy as np

def load_training_labels( filepath=os.path.join( os.getcwd(), 'train_labels.csv' ), as_tensor=False ):
    """
    Wrapper around pds.read_csv with the appropriate args.

    Arguments:

        filepath: path to train_labels.csv 

    Returns:

        pandas DataFrame representation of the file (dtype = int64 ).
    """
    labels = pds.read_csv( 
        filepath, 
        sep=',',
        header=0,
        index_col=None
    )

    if as_tensor:
        return torch.tensor( np.array( labels.Category.values, dtype=np.float32 ) )
    else:
        return labels
    
def load_training_data( filepath=os.path.join( os.getcwd(), 'train_images.pkl' ), as_tensor=False ):
    """
    Wrapper around pickle.load with the appropriate args.

    Arguments:

        filepath: path to train_labels.csv 

    Returns:

        numpy 3D array representation of the file (dtype = float64 ).
    """
    with open( filepath, 'rb' ) as handle:
        data = pickle.load( handle )
    
    if as_tensor:
        return torch.tensor( data )
    
    else:
        return data
'''
class FullTrainingDataset(torch.utils.data.Dataset):
    def init(self, full_ds, offset, length):
    self.full_ds = full_ds
    self.offset = offset
    self.length = length
    assert len(full_ds)>=offset+length, Exception(“Parent Dataset not long enough”)
    super(FullTrainingDataset, self).init()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.full_ds[i+self.offset]

    def trainTestSplit(dataset, val_share=TEST_RATIO):
    val_offset = int(len(dataset)*(1-val_share))
    return FullTrainingDataset(dataset, 0, val_offset), FullTrainingDataset(dataset, val_offset, len(dataset)-val_offset)

    train_ds, val_ds = trainTestSplit(dset_train)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=1,pin_memory=PIN_MEMORY)

    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=1,
    pin_memory=PIN_MEMORY)

'''