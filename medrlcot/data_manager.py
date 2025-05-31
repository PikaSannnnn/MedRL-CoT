import logging
import datasets as hf_datasets
import subprocess
import os
import json

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
    
    data_dir = os.path.join(os.getcwd(), data_dir)
    loaded_datasets = dict()
    for key, dataset in datasets.items():
        ds = None
        os.makedirs(data_dir, exist_ok=True)    # Make data diectory if needed
        if dataset['type'] == 'hf':
            if dataset['src']:  
                ds_all_path = os.path.join(data_dir, key)   # dataset folder to store and load arrow files from
                if os.path.exists(ds_all_path):
                    ds_path = os.path.join(ds_all_path, 'train')
                    if os.path.exists(ds_path):
                        logger.info(f"%s dataset already exists in disk. If the dataset is giving errors or you'd like a fresh install, delete the {ds_path} directory.", dataset['src'])
                        
                        if load:
                            logger.info(f"Loading saved hugginface %s dataset.", dataset['src'])
                            try:
                                # Need list of arrow files in case dataset is split into multiple arrows
                                arrow_dir = os.path.join(ds_path, "train")
                                arrows = [os.path.join(arrow_dir, f) for f in os.listdir(arrow_dir) if f.endswith(".arrow")]
                                ds = hf_datasets.concatenate_datasets([hf_datasets.Dataset.from_file(arrow) for arrow in arrows])
                            except:
                                ds = None
                                logger.error(f"Error loading %s dataset from local!", dataset['src'])
                            else:
                                logger.info(f"Successfully loaded %s as key {key}", dataset['src'])
                    else:
                        logger.info(f"Downloading %s hugging face dataset.", dataset['src'])
                        
                        try:
                            # Download dataset from hf
                            ds = hf_datasets.load_dataset(dataset['src'])
                        except:
                            ds = None
                            logger.error(f"Error downloading %s dataset!", dataset['src'])
                        else:
                            # Save hf dataset as arrow(s)
                            ds.save_to_disk(ds_path)
                            logger.info(f"Done downloading %s and saved to {ds_path}.", dataset['src'])

                # ds_processed_path = os.path.join(ds_all_path, "processed")
                # ds_processed_ckpt = os.path.join(ds_all_path, "checkpoint.json")
                # if os.path.exists(ds_processed_path):
                #     if os.path.exists(ds_processed_dkpt):
                        
                loaded_datasets[key] = ds
            else:
                logger.warning(f"No source was provided for '{key}', skipping...")
        elif dataset['type'] == 'phys':
            if dataset['src']:
                ds_dir = os.path.join(data_dir, key)    # overall dataset directory
                data_path = os.path.join(ds_dir, dataset['src'])    # Path to the csv dataset
                ds_path = os.path.join(ds_dir, "hf")    # path to where the hugginface dataset of the physio csv is saved and loaded from
                if os.path.exists(ds_dir):
                    if os.path.exists(ds_path):
                        logger.info(f"%s dataset already exists in disk as huggin_face dataset. If the dataset is giving errors or you'd like a fresh install, delete the {ds_path} directory.", dataset['src'])
                        
                        if load:
                            logger.info(f"Loading saved hugginface %s dataset.", dataset['src'])
                            try:
                                # Need list of arrow files in case dataset is split into multiple arrows
                                arrow_dir = os.path.join(ds_path, "train")
                                arrows = [os.path.join(arrow_dir, f) for f in os.listdir(arrow_dir) if f.endswith(".arrow")]
                                ds = hf_datasets.concatenate_datasets([hf_datasets.Dataset.from_file(arrow) for arrow in arrows])
                                # ds = hf_datasets.Dataset.from_file(os.path.join(ds_path, "train", "data-00000-of-00001.arrow"))
                            except:
                                ds = None
                                logger.error(f"Error loading %s hugging_face dataset from local!", dataset['src'])
                            else:
                                logger.info(f"Successfully loaded hugging_face of %s as key {key}", dataset['src'])
                    else:
                        try:
                            # Load dataset from pre-downloaded csv
                            ds = hf_datasets.load_dataset(
                                "csv",
                                data_files=f"{data_path}",
                            )
                        except:
                            ds = None
                            logger.error(f"Error loading original %s dataset from local!", dataset['src'])
                        else:
                            # Save to the hugginface location
                            ds.save_to_disk(ds_path)
                            logger.info(f"Done loading %s and saved hugging_face dataset to {ds_path}.", dataset['src'])
                else:
                    logger.error(f"Dataset not found in local. Due to long web-download time, this dataset must be manually installed to local with the dataset expected to be saved as {data_path}.")
    
                # Add dataset to list of datasets if successful (i.e. not None)
                if ds is not None:
                    loaded_datasets[key] = ds
                    
    # Load status
    failed_datasets = set(datasets.keys()) - set(loaded_datasets.keys())
    if load:
        if failed_datasets:
            logger.warning(f"Failed to load {len(failed_datasets)} dataset: {failed_datasets}!")

        logger.info(f"Successfully loaded %d datasets: {list(loaded_datasets.keys())}", len(loaded_datasets))

    return loaded_datasets