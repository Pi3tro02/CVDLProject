import os

DATA_DIR = "/datasets/tdt4265/mic/open/HNTS-MRG"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TRAIN_DIR2 = os.path.join(DATA_DIR, "HNTSMRG24_train")
TEST_DIR = os.path.join(DATA_DIR, "test")

PATCH_SIZE = (128, 128, 64)
BATCH_SIZE = 2
NUM_EPOCHS = 100
LR = 1e-4
VAL_INTERVAL = 1
