from medrlcot.config.env import MedRL_CoT
from medrlcot import data_manager
from medrlcot.medrlcot_logger import setup_logger
from dotenv import load_dotenv
from datasets import Features, Value
from IPython.display import display
import numpy as np
import pandas as pd
import datasets as hf_datasets
import logging
import os
import json

model_cfg_path = os.path.join(os.getcwd(), "medrlcot/config/.env")
medrlcot_config = MedRL_CoT(model_cfg_path)

setup_logger()
preproc_logger = logging.getLogger("MedRL-CoT Preprocess")

classes = np.array(['symptoms_labs', 'thought_process', 'diagnosis'])

def mimic_preprocess(dataset, logger=None):
    if not logger:
        logger = preproc_logger
        
    cleaned_ds = dataset.copy()

    # For loggin purposes
    N = cleaned_ds.shape[0]
    logger.info(f"Found {N} rows")
    num_renames = cleaned_ds[cleaned_ds['class'].isin(['symptoms_lbs', 'symptoms_lads'])].shape[0]
    logger.info(f"Fixed class naming for {num_renames} rows ({(num_renames / N)*100} %)")
    
    # Fix naming of some classes
    cleaned_ds['class'] = cleaned_ds['class'].replace({'symptoms_lbs': 'symptoms_labs', 'symptoms_lads': 'symptoms_labs'})
    
    pos_invalids = cleaned_ds[~cleaned_ds['class'].isin(classes)]
    swapped_values = pos_invalids[pos_invalids['sentence'].str.lower().isin(classes)]    # Rows with swapped values
    
    # Clearly invalids, temp drop to remove from our ceaned list
    invalid_classes = pos_invalids[pos_invalids['class'].str.lower().isin(['', '0', '__', 'None'])]  # collect empty sentence and classes (Note that doing it here will catch the invalid swapped sentences as well)
    invalid_sentences = pos_invalids[pos_invalids['sentence'].str.lower().isin(['', '__', 'None'])]
    invalids = pos_invalids.loc[invalid_classes.index.union(invalid_sentences.index)]
    nonstd_classes = pos_invalids.drop(index=swapped_values.index.union(invalids.index))   # Get list of non-standard classes

    # Get list of classes that can be classified as "other" with enough occurence (non-outliery)
    # value_cnts = nonstd_classes['class'].value_counts()
    # other_classes = value_cnts[value_cnts >= 5].index.tolist()
    # other_class_indices = nonstd_classes[nonstd_classes['class'].isin(other_classes)].index
    
    # Clean the dataset
    swapped_indices = swapped_values.index
    cleaned_ds.loc[swapped_indices, ['sentence', 'class']] = cleaned_ds.loc[swapped_indices, ['class', 'sentence']].values # swap the values in indices where it's swapped
    # cleaned_ds.loc[other_class_indices, 'class'] = 'other'
    # cleaned_ds['class'] = cleaned_ds['class'].apply(lambda x: 'other' if x in other_classes else x)  # relabel non-standards to 'other'
    invalid_sentences = cleaned_ds[cleaned_ds['sentence'].str.lower().isin(['', '__', 'None', '[]', 'False', '()'])]    # Redundant invalid_sentence search in entire cleaned_dataset in case some were uncaught
    drop_indices = cleaned_ds[~cleaned_ds['class'].isin(np.append(classes, 'other'))].index.union(invalid_sentences.index)
    cleaned_ds = cleaned_ds.drop(index=drop_indices) # drop all others that aren't in our list of classes + 'other'  (basically all invalids)

    # Summary
    num_reclass = cleaned_ds[cleaned_ds['class'] == 'other'].shape[0]
    # logger.info(f"Re-classified {len(other_classes)} classes as 'other', or {num_reclass} rows ({(num_reclass / N)*100} %)")
    logger.info(f'Swapped class and sentence values of {swapped_indices.shape[0]} rows ({(swapped_indices.shape[0] / N)*100} %)')
    logger.info(f'Dropped {drop_indices.shape[0]} invalid rows ({(drop_indices.shape[0] / N)*100} %)')

    return cleaned_ds

def aug_preprocess(dataset, logger=None):
    if not logger:
        logger = preproc_logger
        
    cleaned_ds = dataset.copy()

    # For loggin purposes
    N = cleaned_ds.shape[0]
    logger.info(f"Found {N} rows")
    num_renames = cleaned_ds[cleaned_ds['class'].isin(['symptoms_lbs', 'symptoms_lads'])].shape[0]
    logger.info(f"Fixed class naming for {num_renames} rows ({(num_renames / N)*100} %)")
    
    # Fix naming of some classes
    cleaned_ds['class'] = cleaned_ds['class'].replace({'symptoms_lbs': 'symptoms_labs', 'symptoms_lads': 'symptoms_labs'})
    
    pos_invalids = cleaned_ds[~cleaned_ds['class'].isin(classes)]
    swapped_values = pos_invalids[pos_invalids['sentence'].isin(classes)]    # Rows with swapped values
    invalid_classes = pos_invalids[pos_invalids['class'].str.lower().isin(['', '0', '__', 'None', '[]', 'False', 'The', 'No'])]  # collect empty sentence and classes (Note that doing it here will catch the invalid swapped sentences as well)
    ignore_classes = pos_invalids[pos_invalids['sentence'].str.contains('not a sentence')]
    invalid_sentences = pos_invalids[pos_invalids['sentence'].str.lower().isin(['', '__', 'None', '[]', 'False', '()'])]
    
    invalids = pos_invalids.loc[invalid_classes.index.union(invalid_sentences.index.union(ignore_classes.index))]
    # invalids = pos_invalids.loc[invalid_classes.index.union(invalid_sentences.index)]
    nonstd_classes = pos_invalids.drop(index=swapped_values.index.union(invalids.index))   # Get list of non-standard classes

    # Get list of classes that can be classified as "other" with enough occurence (non-outliery)
    value_cnts = nonstd_classes['class'].value_counts()
    other_classes = value_cnts[value_cnts >= 5].index.tolist()
    other_class_indices = nonstd_classes[nonstd_classes['class'].isin(other_classes)].index

    # Clean the dataset
    swapped_indices = swapped_values.index
    cleaned_ds.loc[swapped_indices, ['sentence', 'class']] = cleaned_ds.loc[swapped_indices, ['class', 'sentence']].values # swap the values in indices where it's swapped
    cleaned_ds.loc[other_class_indices, 'class'] = 'other'
    # cleaned_ds['class'] = cleaned_ds['class'].apply(lambda x: 'other' if x in other_classes else x)  # relabel non-standards to 'other'
    invalid_sentences = cleaned_ds[cleaned_ds['sentence'].str.lower().isin(['', '__', 'None', '[]', 'False', '()'])]    # Redundant invalid_sentence search in entire cleaned_dataset in case some were uncaught
    drop_indices = cleaned_ds[~cleaned_ds['class'].isin(np.append(classes, 'other'))].index.union(invalid_sentences.index)
    cleaned_ds = cleaned_ds.drop(index=drop_indices) # drop all others that aren't in our list of classes + 'other' (basically all invalids)

    # Summary
    num_reclass = cleaned_ds[cleaned_ds['class'] == 'other'].shape[0]
    logger.info(f"Re-classified {len(other_classes)} classes as 'other', or {num_reclass} rows ({(num_reclass / N)*100} %)")
    logger.info(f'Swapped class and sentence values of {swapped_indices.shape[0]} rows ({(swapped_indices.shape[0] / N)*100} %)')
    logger.info(f'Dropped {drop_indices.shape[0]} invalid rows ({(drop_indices.shape[0] / N)*100} %)')
    
    return cleaned_ds

def preprocess_datasets(logger=None):
    if not logger:
        logger = preproc_logger
        
    def load_labeled(arrow_dir):
        arrows = [os.path.join(arrow_dir, f) for f in os.listdir(arrow_dir) if f.endswith(".arrow")]
        processed_dataset = hf_datasets.concatenate_datasets([hf_datasets.Dataset.from_file(arrow) for arrow in arrows]).to_dict()

        return processed_dataset
    
    processed_dirs = {ds: os.path.join(os.getcwd(), medrlcot_config.data_dir, ds, 'processed') for ds in medrlcot_config.datasets}
    processed_datasets = {key: pd.DataFrame(load_labeled(processed_dir)) for key, processed_dir in processed_dirs.items()}
    proc_funcs = {'aug_med_notes': aug_preprocess, 'mimic4': mimic_preprocess}
    
    preprocessed_datasets = dict()
    for key, item in processed_datasets.items():
        logger.info("=" * 50)
        logger.info(f"Cleaning up {key} dataset")
        preprocessed_datasets[key] = proc_funcs[key](processed_datasets[key])
        logger.info("=" * 50)
        
    return preprocessed_datasets 


def xy_split_processing_sft(group, x_func=None, y_func=None):
    X = []
    Y = []
    for _, row in group.iterrows():
        if row['class'] == 'symptoms_labs' or row['class'] == 'other':
            X.append(row)
        else:
            Y.append(row)

    X_case = x_func(X) if x_func else ' '.join([str(row['sentence']) for row in X])
    # X_case = ' '.join(f"{row['sentence']} <{row['class']}>" for _, row in group.iterrows()) # Input with all
    Y_case = ' '.join([f"{row['sentence']} <{row['class']}> " for row in Y])    # Output with only thought_process and diagnosis
    
    return pd.Series({'X': X_case, 'Y': Y_case})