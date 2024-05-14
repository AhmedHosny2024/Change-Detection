import os
import time

import numpy as np
from deep_learning_technique.custom_dataset import Custom_dataset as custom_dataset
from deep_learning_technique.config import *
from torch.utils.data import Dataset, DataLoader
from deep_learning_technique.utilties import *
import torch.optim as optim
from deep_learning_technique.generator_copy import Generator
from deep_learning_technique.discriminator import Discriminator

class CDGAN:
    def __init__(self):

        # create a dataloader object
        self.test_dataset=custom_dataset(TEST_FOLDER_PATH,transform_type="test")
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
          total_jaccard_score=[]
          f1_score_list=[]
          for k, data in enumerate(self.test_dataloader):
              x1, x2, label = data
              x1 = x1.to(DEVICE, dtype=torch.float)
              x2 = x2.to(DEVICE, dtype=torch.float)
              label = label.to(DEVICE, dtype=torch.float)
              label = label[:,0,:,:].unsqueeze(1)
              x = torch.cat((x1,x2),1)
              time_i = time.time()
              v_fake = self.netg(x)
              tp, fp, tn, fn = f1_score(v_fake, label)   
              TP += tp
              FN += fn
              TN += tn
              FP += fp
              jaccard_score_,f1_sco= compute_jaccard_index(v_fake,label)
              total_jaccard_score+=jaccard_score_
              f1_score_list+=f1_sco
              save_current_images(label.data, v_fake.data, IM_SAVE_DIR, 'test_output_images',k, self.file_name)
              del x1
              del x2
              del label
              torch.cuda.empty_cache()
          print('TP:',TP)
          print('FP:',FP)
          print('TN:',TN)
          print('FN:',FN)
          precision = TP/(TP+FP+1e-8)
          recall = TP/(TP+FN+1e-8)
          f1 = 2*precision*recall/(precision+recall+1e-8)
          total_jaccard_score=np.mean(total_jaccard_score)
          f1=np.mean(f1_score_list)
          print('test F1: {}'.format(f1))
          print('test jaccard score:{}'.format(total_jaccard_score))   

    def load_model(self):
        self.netg.load_state_dict(torch.load(os.path.join(OUTPUT_PATH, 'netg.pth')))
        print('Model loaded successfully')
        return True
    
if __name__ == "__main__":
    cdgan = CDGAN()
    cdgan.test()
    