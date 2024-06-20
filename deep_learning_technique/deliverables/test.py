import os
import numpy as np
from custom_dataset_test import Custom_dataset_test as custom_dataset
from deep_learning_technique.config import *
from torch.utils.data import  DataLoader
from deep_learning_technique.utilties import *
from deep_learning_technique.generator_copy import Generator

class CDGAN:
    def __init__(self):

        self.test_dataset=custom_dataset(TEST_FOLDER_PATH)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        self.netg = Generator(ISIZE,NC*2, NZ, NDF, EXTRALAYERS).to(DEVICE)
        self.netg.apply(weights_init)
        self.file_name=self.test_dataset.get_files_name()

        
    def test(self):
        print('Testing.................')
        self.load_model()
        self.netg.eval()
        with torch.no_grad(): 
          for k, data in enumerate(self.test_dataloader):
              x1, x2 = data
              x1 = x1.to(DEVICE, dtype=torch.float)
              x2 = x2.to(DEVICE, dtype=torch.float)
              x = torch.cat((x1,x2),1)
              v_fake = self.netg(x)
              save_test_images(v_fake.data, IM_SAVE_DIR, 'test_output_images',k,self.file_name)
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
    