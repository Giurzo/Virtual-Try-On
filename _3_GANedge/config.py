import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
GEN_LEARNING_RATE = 1e-4
DISC_LEARNING_RATE = 1e-5
BATCH_SIZE = 4
NUM_WORKERS = 4
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 500
LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = False
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_DISC = "files/gan/disc.pth.tar"
CHECKPOINT_GEN = "files/gan/gen.pth.tar"
CHECKPOINT_GEN = "files/gan/\BU_08_05\gen.pth.tar"
CHECKPOINT_GEN = "files/gan/gen.pth.tar"