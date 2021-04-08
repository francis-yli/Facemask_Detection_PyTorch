# img_to_tensor
# given classified dataset, resize image and convert to tensor so that could be fed into NN later
# Author: Yangjia Li (Francis)
# Date: Apr. 08, 2021
# Last_Modified: Apr. 08, 2021

import cv2
import numpy as np
from torch import long, tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor


class Img_To_Tensor(Dataset):
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