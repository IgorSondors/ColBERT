import torch

DEVICE = torch.device("cuda")
# DEVICE = torch.device("cpu")

SAVED_CHECKPOINTS = [9555, 9555 * 2, 9555 * 3, 9555 * 4, 9555 * 5, 9555 * 6, 9555 * 7, 9555 * 8, 9555 * 9, 9555 * 10]
# SAVED_CHECKPOINTS += []
SAVED_CHECKPOINTS = set(SAVED_CHECKPOINTS)