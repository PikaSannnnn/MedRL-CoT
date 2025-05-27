from dotenv import load_dotenv
import os

class MedRL_CoT:
    def __init__(self):
        self.env = load_dotenv()
        
        # Meta info
        self.log_dir = os.getenv('log_dir', 'logs')
        self.data_dir = os.getenv('data_dir', 'data')
        
        # Dataset Info
        datasets = [dataset.strip().split('/') for dataset in os.getenv('datasets', '').split(',')]
        self.datasets = {ds_name: {'type': ds_type, 'src': os.getenv(ds_name, None)} for ds_type, ds_name in datasets}
        print(self.datasets)
        
medrlcot_config = MedRL_CoT()