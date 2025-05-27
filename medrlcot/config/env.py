from dotenv import load_dotenv
import os

class MedRL_CoT:
    def __init__(self):
        self.env = load_dotenv()
        
        # Meta info
        self.model_name = os.getenv('model_name', 'MedRl-CoT')
        self.log_name = os.getenv('log_name', 'medrlcot')
        self.log_dir = os.getenv('log_dir', 'logs')
        self.data_dir = os.getenv('data_dir', 'data')
        
        # Dataset Info
        datasets = [dataset.strip().split('/') for dataset in os.getenv('datasets', '').split(',')]
        self.datasets = {ds_name: {'type': ds_type, 'src': os.getenv(ds_name, None)} for ds_type, ds_name in datasets}
        print(self.datasets)
        
medrlcot_config = MedRL_CoT()