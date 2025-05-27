from config.env import medrlcot_config
from config import data_manager
import logging
import os


# Setup logs
os.makedirs(medrlcot_config.log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/medrlcot.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("medrlcot")

# Load datasets
data_manager.load_datasets(medrlcot_config.datasets)