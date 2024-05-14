
# from deep_learning_technique.denoiser.data_loader.custom_dataset_paper import *
from deep_learning_technique.denoiser.config import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys, os, time, argparse, shutil, scipy, h5py, glob
import torchvision.models as models
import mlflow
from deep_learning_technique.denoiser.utils import *
from deep_learning_technique.denoiser.models.gan_model import TomoGAN
from deep_learning_technique.denoiser.options.train_option import TrainOptions
from deep_learning_technique.denoiser.data_loader.custom_dataset import Custom_dataset as custom_dataset
from torch.utils.data import Dataset, DataLoader
from deep_learning_technique.denoiser.config import *
from deep_learning_technique.utilties import *
class DenoiserTrainer():
    def __init__(self):
        self.arg=TrainOptions()
        self.model = TomoGAN(self.arg)
        # self.data_genrator = bkgdGen(data_generator=gen_train_batch_bg(data_file_h5="datasets/train_noise.h5", mb_size=BATCH_SIZE, in_depth=DEPTH, img_size=IMAGE_SIZE), max_prefetch=16)
        self.train_dataset=custom_dataset(TRAIN_DATA,transform_type="train")
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.valid_dataset=custom_dataset(EVAL_DATA,transform_type="test")
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print("train data size",len(self.train_dataset))
        print("valid data size",len(self.valid_dataset))
        self.schedular_gen = optim.lr_scheduler.StepLR(self.model.optimizer_G, step_size=15, gamma=0.5)
        self.schedular_dis = optim.lr_scheduler.StepLR(self.model.optimizer_D, step_size=15, gamma=0.5)

        self.itr_out_dir = NAME + '-itrOut'
        if os.path.isdir(self.itr_out_dir): 
            shutil.rmtree(self.itr_out_dir)
        os.mkdir(self.itr_out_dir) 
        if CONTINUE_TRAIN:
            self.load_model()
    
    def load_model(self):
        self.model.load_models()
    def save_model(self):
        self.model.save_models()

    def train(self):
        best_jaccard_score = 0
        with mlflow.start_run() as run:
            mlflow.log_param("batch_size", BATCH_SIZE)
            mlflow.log_param("depth", DEPTH)
            mlflow.log_param("epochs", EPOCHS)
            mlflow.log_param("image_size", IMAGE_SIZE)
            mlflow.log_param("generator_iterations", ITG)
            mlflow.log_param("discrimiator_iterations", ITD)
            print('[Info] Start training')
            for epoch in range(EPOCHS):
                print('[Info] Epoch: %i' % (epoch))
                total_epochs_loss = 0
                adv_loss = 10000000000
              #TODO:
              # make normal dataloader as its better 
              # it make sure to loop on all trainning data 
              # make generator train many times and then train one time discriminator one time all on same example (batch) 

                for batch_idx,(x1,x2, lable) in enumerate(self.train_dataloader):
                    image=torch.cat((x1,x2),1)
                    X_mb, y_mb = image, lable
                 
                    for _ge in range(ITG):
                        self.model.set_input((X_mb, y_mb))
                        self.model.backward_G()

                    itr_prints_gen = ' Epoch: %i,batch %i, gloss: %.2f (mse%.3f, adv%.3f)' % (\
                    epoch,batch_idx, self.model.loss_G, self.model.loss_G_MSE, self.model.loss_G_GAN, )
                    total_epochs_loss += self.model.loss_G
                      
                      # with open("deep_learning_technique/denoiser/outputs/iter_logs.txt", "w") as f:
                      #   print('%s;' % (itr_prints_gen ))

                    # else :
                    for de in range(ITD):
                        self.model.set_input((X_mb, y_mb))
                        self.model.backward_D()
                  
                    with open("deep_learning_technique/denoiser/outputs/iter_logs.txt", "w") as f:
                        print('%s; dloss: %.2f (r%.3f, f%.3f)' % (itr_prints_gen,\
                        self.model.loss_D, self.model.loss_D_real.detach().cpu().numpy().mean(), self.model.loss_D_fake.detach().cpu().numpy().mean(), \
                        ))
              
                print("average loss for epoch %d is %.2f" % (epoch, total_epochs_loss/len(self.train_dataloader)))
                # if not GEN and adv_loss>self.model.loss_D :
                #     adv_loss = self.model.loss_D
                #     self.save_model()
                # evaluate after finish one epoch
                with torch.no_grad():
                    # X_test, y_test = get1batch4test(data_file_h5=TEST_DATA, in_depth=DEPTH)
                    total_jaccard_score=[]
                    for batch_idx,(x1,x2, lable) in enumerate(self.valid_dataloader):
                        image=torch.cat((x1,x2),1)
                        X_test, y_test = image, lable
                        # move to gpu
                        X_test = X_test.to(self.model.device)
                        y_test = y_test.to(self.model.device)
                        self.model.set_input((X_test, y_test))
                        self.model.forward()
                        pred_img = self.model.fake_C
                        lossMSE = self.model.criterionMSE(pred_img, y_test)
                        lossAdv = self.model.criterionGAN(self.model.netD(pred_img), True)
                        lossG = lossMSE*LMSE  + lossAdv*LADV
                        print('[Info]batch %i Test: gloss: %.2f (mse%.3f, adv%.3f)' % (batch_idx,lossG, lossMSE, lossAdv))
                        jaccard_score_= compute_jaccard_index(pred_img,y_test)
                        total_jaccard_score+=jaccard_score_
                        
                        for i in range(0, X_test.shape[0], 1):
                            # move to cpu
                            y_test = y_test.cpu()
                            pred_img = pred_img.cpu()
                            X_test=X_test.cpu()

                            save2image(y_test[i,0,:,:], '%s/gtruth_%d.png' % (self.itr_out_dir,batch_idx*X_test.shape[0]+ i))
                            save2image(X_test[i,DEPTH//2,:,:], '%s/noisy_%d.png' % (self.itr_out_dir,batch_idx*X_test.shape[0]+ i))
                            save2image(pred_img[i,0,:,:].detach().cpu().numpy(), '%s/pred_%d.png' % (self.itr_out_dir,batch_idx*X_test.shape[0]+ i))

                            mlflow.log_artifacts(self.itr_out_dir)
                        del X_test
                        del y_test
                        torch.cuda.empty_cache()

                    total_jaccard_score=np.mean(total_jaccard_score)
                    if total_jaccard_score > best_jaccard_score: 
                        best_jaccard_score = total_jaccard_score
                        self.save_model()
                    print('[Info] Test: Jaccard Score: %.4f' % (total_jaccard_score))
                    print('[Info] Best Jaccard Score: %.4f' % (best_jaccard_score))

                # if GEN:
                self.schedular_gen.step()
                # else:
                self.schedular_dis.step()
        sys.stdout.flush()

if __name__ == "__main__":
  model= DenoiserTrainer()
  model.train()

  # python -m deep_learning_technique.denoiser.trainer.denoiser_trainer