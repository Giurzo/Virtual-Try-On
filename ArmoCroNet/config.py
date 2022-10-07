import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
LEARNING_RATE = 1e-3
GEN_LEARNING_RATE = 1e-3
DISC_LEARNING_RATE = 1e-3
BATCH_SIZE = 16
NUM_WORKERS = 4
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = True
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "files/enc/disc.pth.tar"
CHECKPOINT_GEN = "files/enc/gen.pth.tar"