import logging
import datasets as hf_datasets
import subprocess
import os
import json

import torch
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset, Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split    # for "even" splitting
from sklearn.utils import shuffle
from transformers import AutoTokenizer
from datetime import datetime

logger = logging.getLogger("DataManager")

default_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

class MRCDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_in=512, max_out=256):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_in = max_in
        self.max_out = max_out

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        in_txt = self.df.loc[idx, 'X']
        out_txt = self.df.loc[idx, 'Y']

        input_enc = self.tokenizer(in_txt, max_length=self.max_in, truncation=True, padding="longest", return_tensors="pt")
        output_enc = self.tokenizer(out_txt, max_length=self.max_out, truncation=True, padding="longest", return_tensors="pt")

        labels = output_enc['input_ids'].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100 

        if idx == 0:
            print("== Input Text ==")
            print(self.tokenizer.decode(input_enc['input_ids'].squeeze(0), skip_special_tokens=True))
            print("\n== Label Text ==")
            print(self.tokenizer.decode([x for x in labels.tolist() if x != self.tokenizer.pad_token_id], skip_special_tokens=True))


        return {
            'input_ids': input_enc['input_ids'].squeeze(0),
            'attention_mask': input_enc['attention_mask'].squeeze(0),
            'labels': labels
            # 'labels': output_enc['input_ids'].squeeze(0)
        }

class MedRL_CoT_Dataset:
    '''
    Custom dataset class meant to store the two datasets and to preprocess them before passing them into dataloaders. 
    
    The dataset is shuffled and split into train, validation, and testing on instantiation. 
    Although if it's not great or want to try with different sets, can call shuffle_split() to do this
    '''
    def __init__(self, datasets, jss=None, split=[.75, .25], seed=None, batch_size=8, tokenizer=None):
        # Meta Init
        self.num_entries = 0
        self.split_ratio = split
        self.seed = seed
        self.batch_size = batch_size
        
        # Dataset Init
        self.original_datasets = datasets
        # self.datasets = self.__tokenize__()
        self.tokenizer = tokenizer if tokenizer else default_tokenizer        
        self.data = jss(self.original_datasets) if jss else self.joint_shuffle_split(seed=seed)
        self.__clean__()

    def __clean__(self):
        for key in self.data:
            dataset = self.__getitem__(key)

            # Delete empty rows
            dataset = dataset[dataset['Y'].str.strip() != '']
            dataset = dataset[dataset['X'].str.strip() != '']
            print(dataset[dataset['Y'].str.strip() == ''].index)
    
            self.data[key] = dataset.transpose()
        
    def __getitem__(self, key):
        # return pd.DataFrame(self.data[key])
        return pd.DataFrame(self.data[key]).transpose()
    
    def __len__(self):
        return len(self.data['train'][0])
        
    def joint_shuffle_split(self, split=None, seed=None):
        if not split:
            split = self.split_ratio
        logger.info(f"Using train-val split: {split}")

        if seed is None:
            seed = np.random.default_rng(int(datetime.now().timestamp())).integers(low=0, high=4294967295)
            
        logger.info(f"Split-Shuffle with seed {seed}")
        
        split_dataset = dict()
        for key, dataset in self.original_datasets.items():
            logger.info(f"Splitting {key} dataset.")
            
            train_split, val_split = train_test_split(dataset, test_size=split[1], random_state=seed, shuffle=True)
            # X_train_tensor = self.tokenizer(train_split['X'].tolist(), padding=True, truncation=True, return_tensors='pt')
            # Y_train_tensor = self.tokenizer(train_split['Y'].tolist(), padding=True, truncation=True, return_tensors='pt')
            # X_val_tensor = self.tokenizer(val_split['X'].tolist(), padding=True, truncation=True, return_tensors='pt')
            # Y_val_tensor = self.tokenizer(val_split['Y'].tolist(), padding=True, truncation=True, return_tensors='pt')
            
            # split_dataset[key] = {'train': [X_train_tensor, Y_train_tensor], 'val': [X_val_tensor, Y_val_tensor]}
            split_dataset[key] = {'train': [train_split['X'], train_split['Y']], 'val': [val_split['X'], val_split['Y']]}
            logger.info(f"Split into {train_split['X'].shape[0]} train rows and {val_split['X'].shape[0]} val rows")
        
        logger.info(f"Creating a single joint dataset of {self.original_datasets.keys()} dataset splits")
        joint_split = {'train': [[], []], 'val': [[], []]}
        for dataset in split_dataset.values():
            for key in ['train', 'val']:
                X_split, Y_split = dataset[key]
                joint_split[key][0].append(X_split)
                joint_split[key][1].append(Y_split)

        # Concatenate into final joint_spit
        for key in ['train', 'val']:
            joint_split[key][0], joint_split[key][1] = shuffle(pd.concat(joint_split[key][0], ignore_index=True), pd.concat(joint_split[key][1], ignore_index=True), random_state=seed+100)
            logger.info(f"Joined {len(joint_split[key][0])} {key} rows")
            logger.info(f"Shuffling {key}'s rows")
            
            joint_split[key][0] = joint_split[key][0].reset_index(drop=True)
            joint_split[key][1] = joint_split[key][1].reset_index(drop=True)

        logger.info(f"Returning shuffled train-val splits")
        return joint_split
    
    def get_torch_dataset(self, key):
        assert key in ['train', 'val']
        return MRCDataset(self.__getitem__(key), self.tokenizer)
        
    
    def get_dataloader(self, key):
        assert key in ['train', 'val']
        key_ds = self.get_torch_dataset(key)
        return DataLoader(key_ds, batch_size=self.batch_size, shuffle=True)
        
    
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
                                arrows = [os.path.join(ds_path, f) for f in os.listdir(ds_path) if f.endswith(".arrow")]
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
                        except Exception as e:
                            ds = None
                            logger.error(f"Error downloading %s dataset!", dataset['src'])
                            logger.error(e)
                        else:
                            # Save hf dataset as arrow(s)
                            ds.save_to_disk(ds_path)
                            logger.info(f"Done downloading %s and saved to {ds_path}.", dataset['src'])
                        
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
                            except Exception as e:
                                ds = None
                                logger.error(f"Error loading %s hugging_face dataset from local!", dataset['src'])
                                logger.error(e)
                            else:
                                logger.info(f"Successfully loaded hugging_face of %s as key {key}", dataset['src'])
                    else:
                        try:
                            # Load dataset from pre-downloaded csv
                            ds = hf_datasets.load_dataset(
                                "csv",
                                data_files=f"{data_path}",
                            )
                        except Exception as e:
                            ds = None
                            logger.error(f"Error loading original %s dataset from local!", dataset['src'])
                            logger.error(e)
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