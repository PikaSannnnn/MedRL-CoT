import logging
import datasets as hf_datasets
import os

logger = logging.getLogger("medrlcot")

def load_datasets(datasets: dict, data_dir: str = 'data'):
    '''
    Load datasets from environment
    '''
    loaded_datasets = dict()
    for key, dataset in datasets.items():
        print(dataset)
        if dataset['type'] == 'hf':
            if dataset['src']:
                os.makedirs(data_dir, exist_ok=True)
                ds_path = os.path.join(data_dir, key)
                if os.path.exists(ds_path):
                    logger.info(f"Loading saved %s dataset from disk. If the dataset is giving errors or you'd like a fresh install, delete the {ds_path} directory.", dataset['src'])
                    
                else:
                    logger.info(f"Downloading %s hugging face dataset.", dataset['src'])
                    
                    ds = hf_datasets.load_dataset(dataset['src'])
                    ds.save_to_disk(ds_path)
                    
                    logger.info(f"Done loading %s and saved to {ds_path}.", dataset['src'])
        elif dataset['type'] == 'phys':
            if dataset['src']:
                print("TODO: Need to load physio dataset")

    return loaded_datasets

def tvt_split():
    '''
    Train-Validation-Test Split
    '''
    pass