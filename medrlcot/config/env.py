from dotenv import load_dotenv
import os

class MedRL_CoT:
    def __init__(self, env=None):
        self.env = load_dotenv() if env is None else load_dotenv(dotenv_path=env)
        # print(env, self.env)
        
        # Meta info
        self.model_name = os.getenv('model_name', 'MedRl-CoT')
        self.log_name = os.getenv('log_name', 'medrlcot')
        self.log_dir = os.getenv('log_dir', 'logs')
        self.data_dir = os.getenv('data_dir', 'data')
        
        print(self.model_name)
        
        # Dataset Info
        datasets = [dataset.strip().split('/') for dataset in os.getenv('datasets', '').split(',')]
        print(datasets)
        self.datasets = {ds_name: {'type': ds_type, 'src': os.getenv(ds_name, None)} for ds_type, ds_name in datasets}
        # print(self.datasets)
        
        # print(os.getcwd())

# medrlcot_config = MedRL_CoT()