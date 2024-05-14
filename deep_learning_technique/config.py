import torch


TRAIN_FOLDER_PATH = 'trainval/train'
TEST_FOLDER_PATH = 'trainval/test'
VAL_FOLDER_PATH = 'trainval/validation'
OUTPUT_PATH = 'deep_learning_technique/output'
IM_SAVE_DIR ='deep_learning_technique/output'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# training configuration
EPOCH = 20
ISIZE = 256         # input image size
RESUME_TRAINING = True
BATCH_SIZE = 6 
DISPLAY = True      # if display training phase in Visdom
DISPOLAY_STEP = 20    
RESUME = False       # if resume from the last epoch

# evaluation configuration
THRESHOLD = 0.5
SAVE_TEST_IAMGES = True # if save change maps during test

# optimizer configuration
LR = 0.000002        # learning rate
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
LR_STEP_SIZE = 10
GAMMA = 0.5
# G_WEIGHT = 200    # loss weight
G_WEIGHT = 200    # loss weight
D_WEIGHT = 1        # loss weight
L1_WEIGHT = 1    # target 1 weight
L0_WEIGHT = 1     # target 0 weight

# networks configuration
NC = 3      # input image channel size 
NZ = 1024        # size of the latent z vector
NDF = 64        # the dimension size of the first convolutional of the generator
NGF = 64        # the dimension size of the first convolutional of the discriminator
EXTRALAYERS = 3 # add extral layers for the generator and discriminator
Z_SIZE = 16
GT_C = 1    # the channel size of ground truth