import json 
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.utils.data as data

 # Dict for changing labels strings to id 
label_dict = {
   'Blouse': 0,
    'Dress': 1,
    'Jeans': 2,
    'Skirt': 3,
     'Tank': 4    
    }
 
class ImageDataset(data.Dataset):

    def __init__(self, filename, transform=None):
        with open(filename, 'r') as f:
            data = json.load(f)
 
        df = pd.DataFrame(data).values
 
        self.transform = transform
        self.file_name = filename.split('.')[0]
        self.data = df[:,1]
        self.n_samples = self.data.shape[0]
        labels = df[:,0]
        labels_index = [label_dict[label] for label in labels] # Changing strings in labels to int by comparision with dict
        array = np.zeros((len(labels), 5), dtype='f')

        for i in range(len(labels_index)):
        	array[i][labels_index[i]] = 1

        self.target = torch.from_numpy(np.array(labels_index)).long()
 		
   
    def __len__(self):  
        return self.n_samples
   
    def __getitem__(self, index):
        img = Image.open("{}/{}".format(self.file_name, self.data[index]))
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, self.target[index]
