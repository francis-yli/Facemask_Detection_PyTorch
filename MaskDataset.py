# MaskDataset
# given classified dataset, resize images and convert to set of tensors
# which is the form accepted by PyTorch NN
# Author: Yangjia Li (Francis)
# Date: Apr. 08, 2021
# Last_Modified: Apr. 08, 2021

import cv2
import numpy as np
from torch import long, tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

class MaskDataset(Dataset):
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame
        
        self.transformations = Compose([
            ToPILImage(),
            Resize((100, 100)),
            ToTensor(), # [0, 1]
        ])
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            raise NotImplementedError('slicing is not supported')
        
        row = self.dataFrame.iloc[key]
        image = cv2.imdecode(np.fromfile(row['image'], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        return {
            'image': self.transformations(image),
            'mask': tensor([row['mask']], dtype=long), # pylint: disable=not-callable
        }
    
    def __len__(self):
        return len(self.dataFrame.index)

def 


def prepare_data(self) -> None:
    self.maskDF = maskDF = pd.read_pickle(self.maskDFPath)
    train, validate = train_test_split(maskDF, test_size=0.3, random_state=0,
                                       stratify=maskDF['mask'])
    self.trainDF = MaskDataset(train)
    self.validateDF = MaskDataset(validate)