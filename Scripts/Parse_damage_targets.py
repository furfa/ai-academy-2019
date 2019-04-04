from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy
import sys
sys.path.append("..")
import os
pd.set_option("max_columns", 10000)

import warnings
warnings.filterwarnings('ignore')

import json

import copy

from Scripts.saving import submit

from Scripts.utils import *

import json

def parse(damage_targets):
    all_targets = set()
    for dt in damage_targets:
        dt = json.loads(dt)
        all_targets = all_targets | set(dt.keys())
    
    print("Len uniq keys", len(all_targets) )

    zeros_dict = {name:0 for name in all_targets}
    new_data = pd.DataFrame(columns = list(all_targets) )

    for dt, i in tqdm(
                    zip(damage_targets, damage_targets.index), 
                    total= damage_targets.shape[0]
                    ):
        dt = json.loads(dt)
        
        nd =  {
                        **zeros_dict,
                        **dt
                    }

        new_data.loc[i] = nd      
        
    print(new_data.shape)

    return new_data

def main():
    base_dir = ".."
    input_dir = open( os.path.join(base_dir, "datadir.txt"), "r").read()[:-1]
    pred_dir = os.path.join(base_dir, "predictions")
    processed_dir = os.path.join(input_dir, "processed")

    print("reading")
    train = pd.read_csv( os.path.join(input_dir,"processed", "train_all_JSON.csv"), index_col=0 )
    test = pd.read_csv( os.path.join(input_dir,"processed", "test_all_JSON.csv"), index_col=0 )

    train = train.damage_targets
    test = test.damage_targets

    print("TRAIN")

    parse(train).to_csv(
        os.path.join(processed_dir, "train_damage_targets.csv")
    )

    print("TEST")  

    parse(test).to_csv(
        os.path.join(processed_dir, "test_damage_targets.csv")
    )

if __name__ == "__main__":
    main()