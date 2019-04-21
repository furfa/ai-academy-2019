from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy
import sys
sys.path.append("..")
import os
pd.set_option("max_columns", 10000)

import json

from tqdm import tqdm

import copy

from utils import *


if __name__ == "__main__":
    base_dir = ".."
    input_dir = open( os.path.join(base_dir, "datadir.txt"), "r").read()[:-1]
    pred_dir = os.path.join(base_dir, "predictions")
    processed_dir = os.path.join(input_dir, "processed")

    items = pd.read_csv( os.path.join(input_dir, 'dota2_items.csv'), index_col=0 )
    items = items.drop(["notes","dname"], axis=1)
    items.qual = items.qual.fillna("nan_str")
    encode_columns(items, ["qual"])

    items.to_csv(
        os.path.join(input_dir, "processed", 'items_proc.csv')
    )
