import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 192
IMAGE_CHANNEL = 3