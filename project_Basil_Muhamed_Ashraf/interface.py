
from model import WasteClassifier as TheModel
from train import train_model_no_val as the_trainer
from predict import classify_waste as the_predictor
from dataset import WasteDataset as TheDataset
from dataset import WasteDataLoader as the_dataloader
from config import BATCH_SIZE as the_batch_size
from config import NUM_EPOCHS as total_epochs