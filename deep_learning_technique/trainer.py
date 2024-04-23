import os
import time

import numpy as np
from deep_learning_technique.custom_dataset import Custom_dataset as custom_dataset
from deep_learning_technique.config import *
from torch.utils.data import Dataset, DataLoader
from deep_learning_technique.utilties import *
import torch.optim as optim
from deep_learning_technique.generator import Generator
from deep_learning_technique.discriminator import Discriminator

class CDGAN:
    def __init__(self):

        # create a dataloader object
        self.train_dataset=custom_dataset(TRAIN_FOLDER_PATH)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=True)
        self.valid_dataset=custom_dataset(VAL_FOLDER_PATH)
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=1, shuffle=True)
        #Models
        self.netg = Generator(ISIZE,NC*2, NZ, NDF, EXTRALAYERS).to(DEVICE)
        self.netd = Discriminator(ISIZE, GT_C, 1, NGF, EXTRALAYERS).to(DEVICE)
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)
        #Losses
        # self.l_adv = l2_loss
        self.l_con = nn.L1Loss()
        # self.l_enc = l2_loss
        self.l_bce = nn.BCELoss()
        # self.l_cos = cos_loss
        # self.dice = DiceLoss()
        self.optimizer_d = optim.Adam(self.netd.parameters(), lr=LR, betas=(0.5, 0.999))
        self.optimizer_g = optim.Adam(self.netg.parameters(), lr=LR, betas=(0.5, 0.999))

    def train(self):
        init_epoch = 0
        best_f1 = 0
        total_steps = 0
        start_time = time.time()
        for epoch in range(init_epoch+1, EPOCH):
            loss_g = []
            loss_d = []
            self.netg.train()
            self.netd.train()
            epoch_iter = 0
            for i, data in enumerate(self.train_dataloader ):
                INPUT_SIZE = [ISIZE,ISIZE] 
                x1, x2, gt = data
                x1 = x1.to(DEVICE, dtype=torch.float)
                x2 = x2.to(DEVICE, dtype=torch.float)
                gt = gt.to(DEVICE, dtype=torch.float)
                gt = gt[:,0,:,:].unsqueeze(1)
                x = torch.cat((x1,x2),1)
                
                epoch_iter +=BATCH_SIZE
                total_steps += BATCH_SIZE
                real_label = torch.ones (size=(x1.shape[0],), dtype=torch.float32, device=DEVICE)
                fake_label = torch.zeros(size=(x1.shape[0],), dtype=torch.float32, device=DEVICE)
                
                #forward
                fake = self.netg(x)
                pred_real = self.netd(gt)
                pred_fake = self.netd(fake).detach()
                err_d_fake = self.l_bce(pred_fake, fake_label)
                err_g = self.l_con(fake, gt)
                err_g_total = G_WEIGHT*err_g +D_WEIGHT*err_d_fake
                
                pred_fake_ = self.netd(fake.detach())
                err_d_real = self.l_bce(pred_real, real_label)
                err_d_fake_ = self.l_bce(pred_fake_, fake_label)
                err_d_total = (err_d_real + err_d_fake_) * 0.5
                
                #backward
                self.optimizer_g.zero_grad()
                err_g_total.backward(retain_graph = True)
                self.optimizer_g.step()
                self.optimizer_d.zero_grad()
                err_d_total.backward()
                self.optimizer_d.step()
                
                errors =get_errors(err_d_total, err_g_total)            
                loss_g.append(err_g_total.item())
                loss_d.append(err_d_total.item())
                
            #     counter_ratio = float(epoch_iter) / len(train_dataloader.dataset)
            #     if(i%ct.DISPOLAY_STEP==0 and i>0):
            #         print('epoch:',epoch,'iteration:',i,' G|D loss is {}|{}'.format(np.mean(loss_g[-51:]),np.mean(loss_d[-51:])))
            #         if ct.DISPLAY:
            #             utils.plot_current_errors(epoch, counter_ratio, errors,vis)
            #             utils.display_current_images(gt.data, fake.data, vis)
            # utils.save_current_images(epoch, gt.data, fake.data, ct.IM_SAVE_DIR, 'training_output_images')
            
            with open(os.path.join(OUTPUT_PATH,'train_loss.txt'),'a') as f:
                f.write('after %s epoch, loss is %g,loss1 is %g,loss2 is %g,loss3 is %g'%(epoch,np.mean(loss_g),np.mean(loss_d),np.mean(loss_g),np.mean(loss_d)))
                f.write('\n')
            # if not os.path.exists(ct.WEIGHTS_SAVE_DIR):
            #     os.makedirs(ct.WEIGHTS_SAVE_DIR)
            # utils.save_weights(epoch,netg,optimizer_g,ct.WEIGHTS_SAVE_DIR, 'netG')
            # utils.save_weights(epoch,netd,optimizer_d,ct.WEIGHTS_SAVE_DIR, 'netD')
            duration = time.time()-start_time
            print('training duration is %g'%duration)



            #val phase
            print('Validating.................')
            # pretrained_dict = torch.load(os.path.join(ct.WEIGHTS_SAVE_DIR,'current_netG.pth'))['model_state_dict']
            # DEVICE = torch.DEVICE('cuda' if torch.cuda.is_available() else 'cpu')
            # net = NetG(ct.ISIZE, ct.NC*2, ct.NZ, ct.NDF, ct.EXTRALAYERS).to(DEVICE=DEVICE)
            # net.load_state_dict(pretrained_dict,False)
            with self.netg.eval() and torch.no_grad(): 
                TP = 0
                FN = 0
                FP = 0
                TN = 0
                for k, data in enumerate(self.valid_dataloader):
                    x1, x2, label = data
                    x1 = x1.to(DEVICE, dtype=torch.float)
                    x2 = x2.to(DEVICE, dtype=torch.float)
                    label = label.to(DEVICE, dtype=torch.float)
                    label = label[:,0,:,:].unsqueeze(1)
                    x = torch.cat((x1,x2),1)
                    time_i = time.time()
                    v_fake = self.netg(x)
                    
                    tp, fp, tn, fn = f1_score(v_fake, label)   
                    print("f1 score: ",tp, fp, tn, fn) 
                    TP += tp
                    FN += fn
                    TN += tn
                    FP += fp
                
                precision = TP/(TP+FP+1e-8)
                oa = (TP+TN)/(TP+FN+TN+FP+1e-8)
                recall = TP/(TP+FN+1e-8)
                f1 = 2*precision*recall/(precision+recall+1e-8)
                # if not os.path.exists(BEST_WEIGHT_SAVE_DIR):
                #     os.makedirs(ct.BEST_WEIGHT_SAVE_DIR)
                if f1 > best_f1: 
                    best_f1 = f1
                #     shutil.copy(os.path.join(ct.WEIGHTS_SAVE_DIR,'current_netG.pth'),os.path.join(ct.BEST_WEIGHT_SAVE_DIR,'netG.pth'))           
                print('current F1: {}'.format(f1))
                print('best f1: {}'.format(best_f1))
                with open(os.path.join(OUTPUT_PATH,'f1_score.txt'),'a') as f:
                    f.write('current epoch:{},current f1:{},best f1:{}'.format(epoch,f1,best_f1))
                    f.write('\n')  
    
if __name__ == "__main__":
    cdgan = CDGAN()
    cdgan.train()
    