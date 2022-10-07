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
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT = "files/beauty/model_no_norm.pth"
CHECKPOINT_GEN = "files/beauty/model_gen.pth"
CHECKPOINT_DIS = "files/beauty/model_disc.pth"