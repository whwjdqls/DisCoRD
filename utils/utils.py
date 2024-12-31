import os, random
import torch
import numpy as np

def generate_date_time():
    """
    Generate a compact date and time string for logging
    :return: date and time in compact format
    """
    import datetime
    now = datetime.datetime.now()
    # Compact format: MMDDHHMMSS
    date_time = now.strftime("%m%d%H%M%S")
    return date_time

def seed_everything(seed: int): 
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
    # Disabling the benchmarking feature with torch.backends.cudnn.benchmark = False causes 
    # cuDNN to deterministically select an algorithm, possibly at the cost of reduced performance.
    # torch.backends.cudnn.benchmark = False # -> Might want to set this to True if it's too slow

def compare_checkpoint_model(checkpoint, model):
    """
    Compare the model with the checkpoint
    :param checkpoint: tcheckpoint
    :param model: the model
    :return: True if the model is the same as the checkpoint
    """
    model_dict = model.state_dict()
    if len(checkpoint) != len(model_dict): # check length
        return False
    
    for key in checkpoint:
        if key not in model_dict: # check key
            raise ValueError(f"Checkpoint key {key} is not in model_dict")
        if checkpoint[key].shape != model_dict[key].shape: # check shape
            raise ValueError(f"Checkpoint shape {checkpoint[key].shape} is not the same as model shape {model_dict[key].shape}")
        if checkpoint[key].dtype != model_dict[key].dtype: # check dtype
            raise ValueError(f"Checkpoint dtype {checkpoint[key].dtype} is not the same as model dtype {model_dict[key].dtype}")
        if not torch.equal(checkpoint[key], model_dict[key]): # check value
            raise ValueError(f"Checkpoint {checkpoint[key]} is not the same as model {model_dict[key]}")
    
    for key in model_dict:
        if key not in checkpoint:
            raise ValueError(f"Model key {key} is not in checkpoint")
        if checkpoint[key].shape != model_dict[key].shape:
            raise ValueError(f"Checkpoint shape {checkpoint[key].shape} is not the same as model shape {model_dict[key].shape}")
        if checkpoint[key].dtype != model_dict[key].dtype:
            raise ValueError(f"Checkpoint dtype {checkpoint[key].dtype} is not the same as model dtype {model_dict[key].dtype}")
        if not torch.equal(checkpoint[key], model_dict[key]):
            raise ValueError(f"Checkpoint {checkpoint[key]} is not the same as model {model_dict[key]}")
         
    return True

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list or 'diffloss' in name:
            no_decay.append(param)  # no weight decay on bias, norm and diffloss
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]