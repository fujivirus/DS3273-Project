# Contains hyperparameters for training such as number_of_epochs, batch_size, etc.
# Also lists the image resize info (images of size (128, 128, 3) = (resize_x, resize_y, input_channels))
# dataset.py file (or wherever resizing is done) imports these variables to do the resizing.

"""
Configuration file for training, dataset, and model hyperparameters.
"""

from torch.cuda import is_available

# Directories
DATA_DIR = "data/images"
MODEL_SAVE_PATH = "checkpoints/best_model.pth"

# Hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Image dimensions
RESIZE_X = 128
RESIZE_Y = 128
INPUT_CHANNELS = 3

# Other
DEVICE = "cuda" if is_available() else "cpu"

# Number of classes
NUM_CLASSES = 30

# Number of DataLoader workers
NUM_WORKERS = 8