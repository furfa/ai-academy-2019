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

base_dir = ".."
input_dir = open( os.path.join(base_dir, "datadir.txt"), "r").read()[:-1]
pred_dir = os.path.join(base_dir, "predictions")
processed_dir = os.path.join(input_dir, "processed")

abilities = pd.read_csv( os.path.join(input_dir, 'dota2_abilities.csv') ).drop(["dname", "desc"], axis=1).set_index("ability_id")
def parse_abilities_behavior(x):
    x = x.strip("[").strip("]")
    splited = x.split(",")
    
    def prep(x):
        x = x.strip(" ").strip("'")
        
        if x == "None":
            return np.nan
        elif x == 'nan':
            return np.nan
        return x
    splited = [prep(s) for s in splited]
    return splited
abilities.behavior = abilities.behavior.apply(parse_abilities_behavior)

unique_behaviors = set()
for b in abilities.behavior:
    unique_behaviors = unique_behaviors | set(b)

def parse(ability_upgrades):

    def process_row(row):
        behavior_counter = {name:0 for name in unique_behaviors}
        for i in row:
            for beh in abilities.ix[ i ].values[0]:
                behavior_counter[beh] += 1
        return behavior_counter
    
    new_data = pd.DataFrame(columns=unique_behaviors)
    for row, i in tqdm( zip(ability_upgrades, ability_upgrades.index), total=ability_upgrades.shape[0] ):
        new_data.loc[i] = process_row( json.loads(row) )

    return new_data

def main():
    base_dir = ".."
    input_dir = open( os.path.join(base_dir, "datadir.txt"), "r").read()[:-1]
    pred_dir = os.path.join(base_dir, "predictions")
    processed_dir = os.path.join(input_dir, "processed")

    print("reading")
    train = pd.read_csv( os.path.join(input_dir,"processed", "train_all_JSON.csv"), index_col=0 )
    test = pd.read_csv( os.path.join(input_dir,"processed", "test_all_JSON.csv"), index_col=0 )

    train = train.ability_upgrades
    test = test.ability_upgrades

    print("TRAIN")

    parse(train).to_csv(
        os.path.join(processed_dir, "train_ability_upgrades.csv")
    )

    print("TEST")  

    parse(test).to_csv(
        os.path.join(processed_dir, "test_ability_upgrades.csv")
    )

if __name__ == "__main__":
    main()