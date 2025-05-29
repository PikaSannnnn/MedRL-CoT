from medrlcot.config.env import MedRL_CoT
from medrlcot import data_manager
from medrlcot.medrlcot_logger import setup_logger
from dotenv import load_dotenv
import logging
import os

# env = load_dotenv()
# model_cfg_path = os.path.join(os.getcwd(), os.getenv('model_config'))
model_cfg_path = os.path.join(os.getcwd(), "medrlcot/config/.env")
medrlcot_config = MedRL_CoT(model_cfg_path)

setup_logger()
logger = logging.getLogger("MedRL-CoT Setup")

# Download datasets
datasets = data_manager.load_datasets(medrlcot_config.datasets, data_dir=medrlcot_config.data_dir)


# print(datasets['aug_med_notes']['full_note'][0])
# data_manager.load_datasets(medrlcot_config.datasets, data_dir=medrlcot_config.data_dir, load=False)