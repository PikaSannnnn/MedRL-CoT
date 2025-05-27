from medrlcot.config.env import medrlcot_config
from medrlcot import data_manager
from medrlcot.medrlcot_logger import setup_logger
import logging

setup_logger()
logger = logging.getLogger("MedRL-CoT Setup")

# Download datasets
data_manager.load_datasets(medrlcot_config.datasets, data_dir=medrlcot_config.data_dir, load=False)