class _RM:
    def __init__(self):
        pass

class _DGM:
    def __init__(self):
        pass
    
    def train(Self):
        pass

class MedRL_CoT:
    def __init__(self):
        self.model = None
    
    def train(self):
        pass

# def join_sentence_class(group):
#     return ' '.join(f"{row['sentence']} <{row['class']}>" for _, row in group.iterrows())

if __name__ == "__main__":
    import medrlcot.preprocessing as mp
    preprocessed_datasets = mp.preprocess_datasets()
    
    # Combine cases into one for cases as example for SFT
    cases_datasets = dict()
    for key, dataset in preprocessed_datasets.items():
        cases_datasets[key] = dataset.groupby('case_id').apply(mp.xy_split_processing).reset_index().sort_values('case_id')
    
    import os
    from medrlcot import data_manager
    import medrlcot.config.env as mce
    model_cfg_path = os.path.join(os.getcwd(), "medrlcot/config/.env")
    medrlcot_config = mce.MedRL_CoT(model_cfg_path)
    raw_datasets = data_manager.load_datasets(medrlcot_config.datasets, data_dir=medrlcot_config.data_dir)  # Load raw dataset (original cases) for RM
    
    # Create customd dataset obj
    cases_data = data_manager.Dataset(cases_datasets)