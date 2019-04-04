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

items = pd.read_csv( os.path.join(input_dir,"processed", 'items_proc.csv'), index_col=0 )

zeros_dict = {i : 0 for i in items.index}

def parse_row(row):
    first_buy_time = dict()
    count_items = dict()
    row = json.loads(row)
    for pair in row[::-1]:
        first_buy_time[ pair["item_id"] ] = pair["time"]
        try:
            count_items[ pair["item_id"] ] += 1
        except KeyError:
            count_items[ pair["item_id"] ] = 1
    
    return {**zeros_dict, **first_buy_time}, {**zeros_dict, **count_items}

def parse(log):
    df_fbt = pd.DataFrame(columns=list(items.index))
    df_ci = pd.DataFrame(columns=list(items.index))
    for row, i in tqdm( zip(log, log.index), total=log.shape[0] ):
        df_fbt.loc[i], df_ci.loc[i] = parse_row(row)
    return pd.concat(
                (
                    df_fbt.add_prefix("first_buy_time_"),
                    df_ci.add_prefix("count_item_"),
                ), axis=1
            )

def main():
    base_dir = ".."
    input_dir = open( os.path.join(base_dir, "datadir.txt"), "r").read()[:-1]
    pred_dir = os.path.join(base_dir, "predictions")
    processed_dir = os.path.join(input_dir, "processed")

    print("reading")
    train = pd.read_csv( os.path.join(input_dir,"processed", "train_all_JSON.csv"), index_col=0 )
    test = pd.read_csv( os.path.join(input_dir,"processed", "test_all_JSON.csv"), index_col=0 )

    train = train.item_purchase_log
    test = test.item_purchase_log

    print("TRAIN")

    parse(train).to_csv(
        os.path.join(processed_dir, "train_item_purchase_log.csv")
    )

    print("TEST")  

    parse(test).to_csv(
        os.path.join(processed_dir, "test_item_purchase_log.csv")
    )

if __name__ == "__main__":
    main()