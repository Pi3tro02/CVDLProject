import os

DATA_DIR = "/cluster/projects/vc/data/mic/open/HNTS-MRG"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TRAIN_DIR2 = os.path.join(DATA_DIR, "HNTSMRG24_train")
TEST_DIR = os.path.join(DATA_DIR, "test")

PATCH_SIZE = (96, 96, 32)
BATCH_SIZE = 2
NUM_EPOCHS = 100
LR = 2e-4
