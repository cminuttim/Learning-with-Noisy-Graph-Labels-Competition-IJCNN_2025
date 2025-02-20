# source/utils.py
import random
import torch
import numpy as np
import logging
from typing import Dict, Any

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(model: torch.nn.Module, 
                   folder_name: str,
                   cycle: int,
                   epoch: int,
                   metrics: Dict[str, Any]):
    filename = f"checkpoints/model_{folder_name}_cycle_{cycle}_epoch_{epoch}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'cycle': cycle,
        'epoch': epoch
    }, filename)
    logging.info(f"Saved checkpoint: {filename}")