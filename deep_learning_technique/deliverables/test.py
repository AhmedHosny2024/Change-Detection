import os
import time

import numpy as np
from deep_learning_technique.deliverables.custom_dataset_test import Custom_dataset_test as custom_dataset
from deep_learning_technique.config import *
from torch.utils.data import Dataset, DataLoader
from deep_learning_technique.utilties import *
import torch.optim as optim
from deep_learning_technique.generator import Generator

class CDGAN:
    def __init__(self):

        # create a dataloader object
        self.test_dataset=custom_dataset(TEST_FOLDER_PATH)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        #Models
        self.file_name=self.test_dataset.get_files_name()
        self.netg = Generator(ISIZE,NC*2, NZ, NDF, EXTRALAYERS).to(DEVICE)
        self.netg.apply(weights_init)
        
    def test(self):
        print('Testing.................')
        self.load_model()
        self.netg.eval()
        with torch.no_grad(): 
          TP = 0
          FN = 0
          FP = 0
          TN = 0
          time_i = time.time()
          for k, data in enumerate(self.test_dataloader):
              x1, x2 = data
              x1 = x1.to(DEVICE, dtype=torch.float)
              x2 = x2.to(DEVICE, dtype=torch.float)
              x = torch.cat((x1,x2),1)
              v_fake = self.netg(x)
              save_test_images(v_fake.data, IM_SAVE_DIR, 'test_output_images',k)
              del x1
              del x2
              torch.cuda.empty_cache()  

    def load_model(self):
        self.netg.load_state_dict(torch.load(os.path.join(OUTPUT_PATH, 'netg.pth')))
        print('Model loaded successfully')
        return True
    
if __name__ == "__main__":
    cdgan = CDGAN()
    cdgan.test()
    