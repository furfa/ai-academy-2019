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

    heroes = pd.read_csv( os.path.join(input_dir, 'dota2_heroes.csv'))

    heroes = heroes.drop(['name', 'localized_name'],axis=1).fillna(0)

    encode_columns(heroes, ["attack_type", "primary_attr"])

    heroes = pd.concat(
        [
            heroes.drop(["roles"], axis=1),
            onehot_lists(heroes.roles),
        ], axis=1
    )

    heroes.to_csv(
        os.path.join(input_dir, "processed", 'heroes_proc.csv')
    )
