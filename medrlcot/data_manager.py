import logging
import datasets as hf_datasets
import os

import torch
from torch.utils.data import DataLoader
import numpy as np

logger = logging.getLogger("DataManager")

class Dataset:
    '''
    Custom dataset class meant to store the two datasets and to preprocess them before passing them into dataloaders. 
    
    The dataset is shuffled and split into train, validation, and testing on instantiation. 
    Although if it's not great or want to try with different sets, can call shuffle_split() to do this
    '''
    def __init__(self, dataset, process_func, split_keys=['train', 'val', 'test'], split=[70, 20, 10]):
        # Meta Init
        self.num_entries = 0
        self.dataset_keys = split_keys
        self.split_ratio = split
        self.split_indices = []
        # TODO: Compute split indices
        
        # Dataset Init
        self.full_dataset = dataset if isinstance(dataset, torch.utils.data.Dataset) else self.__ds_to_tensor__(dataset)
        self.dataset = {key: None for key in split_keys}
        
        # Preprocess data into the note types
        self.__preprocess__(self, process_func=process_func)
        
        # Initial shuffle split
        self.shuffle_split()
            
    def __ds_to_tensor__(self, dataset):
        if isinstance(dataset, hf_datasets.Dataset):
            dataset.set_format("torch")
        else:
            logger.critical("Unsupported dataset type")
            raise TypeError("Unsupported dataset type")
        
        return dataset
        
    def __preprocess__(self, process_func):
        '''
        Call this func to process the data. Mainly preprocessing so we can split the medical notes of the two different datasets separately into its respective note types
        Should instiantiate the function witht he model outside of the class.
        '''
        
        pass
    
    def shuffle_split(self, split=None):
        if not split:
            split = self.split_ratio
        
        # TODO: Shuffle and split into the different trainin datas
        pass
    
def load_datasets(datasets: dict, data_dir: str = 'data', load: bool = True) -> dict:
    '''
    Load datasets from environment
    '''
    logger.info(f"Loading datasets: {list(datasets.keys())}")
    
    loaded_datasets = dict()
    for key, dataset in datasets.items():
        ds = None
        if dataset['type'] == 'hf':
            if dataset['src']:
                os.makedirs(data_dir, exist_ok=True)
                ds_path = os.path.join(data_dir, key)
                if os.path.exists(ds_path):
                    logger.info(f"%s dataset already exists in disk. If the dataset is giving errors or you'd like a fresh install, delete the {ds_path} directory.", dataset['src'])
                    
                    if load:
                        logger.info(f"Loading %s dataset.", dataset['src'])
                        logger.critical("TODO: NEED TO IMPLEMENT HF DATASET LOAD FROM DISK")
                        logger.info(f"Successfully loaded %s as key {key}", dataset['src'])
                else:
                    logger.info(f"Downloading %s hugging face dataset.", dataset['src'])
                    
                    ds = hf_datasets.load_dataset(dataset['src'])
                    ds.save_to_disk(ds_path)
                    
                    logger.info(f"Done downloading %s and saved to {ds_path}.", dataset['src'])
                
                loaded_datasets[key] = ds
            else:
                logger.warning(f"No source was provided for '{key}', skipping...")
        elif dataset['type'] == 'phys':
            if dataset['src']:
                os.makedirs(data_dir, exist_ok=True)
                ds_path = os.path.join(data_dir, key)
                if os.path.exists(ds_path):
                    logger.info(f"%s dataset already exists in disk. If the dataset is giving errors or you'd like a fresh install, delete the {ds_path} directory.", dataset['src'])
                    
                    if load:
                        logger.info(f"Loading %s dataset.", dataset['src'])
                        logger.critical("TODO: NEED TO IMPLEMENT HF DATASET LOAD FROM DISK")
                        logger.info(f"Successfully loaded %s as key {key}", dataset['src'])
                else:
                    logger.critical("TODO: NEED TO IMPLEMENT PHYSIO DATASET LOADING")
            else:
                logger.warning(f"No source was provided for '{key}', skipping...")
    
    failed_datasets = set(datasets.keys()) - set(loaded_datasets.keys())
    if load:
        if failed_datasets:
            logger.warning(f"Failed to load {len(failed_datasets)} dataset: {failed_datasets}!")

        logger.info(f"Successfully loaded %d datasets: {list(loaded_datasets.keys())}", len(loaded_datasets))

    return loaded_datasets