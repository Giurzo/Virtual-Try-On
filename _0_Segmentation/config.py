import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
LEARNING_RATE_GEN = 1e-4
LEARNING_RATE_DIS = 1e-5
BATCH_SIZE = 4
NUM_WORKERS = 4
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 192
IMAGE_CHANNEL = 3
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT = "files/seg/model_no_gan.pth"
CHECKPOINT_GEN = "files/seg/model_gen.pth"
CHECKPOINT_DIS = "files/seg/model_disc.pth"