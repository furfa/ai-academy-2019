from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy
import sys
sys.path.append("..")
import os
pd.set_option("max_columns", 10000)

%pylab inline
import seaborn as sns

import warnings
warnings.simplefilter("ignore")

import json

from tqdm import tqdm

import copy

from Scripts.saving import submit
from utils import linreg_trend, make_diff_shifts


def get_data(data_json_series, shifts, func_to_aggregate):


    data_series = {
        f"{it}->{s}_{func.__name__}":list() for func in func_to_aggregate 
                            for it in ("radiant", "dire", "time", "player")
                            for s in shifts
        
    }
    data_series["id"] = list()

    print(f"making df with shape : {len(data_series.keys())}")

    for ind in tqdm(data_json_series.index):
        ser = json.loads(
            data_json_series[ind]
        )
        
        data_series["id"].append(ind)
        
        sootv = {
            "radiant" : np.array(ser["radiant_gold"]),
            "dire" : np.array(ser["dire_gold"]),
            "time" : np.array(ser["time"]),
            "player" : np.array(ser["player_gold"]),
        }
        
        for it in ("radiant", "dire", "time", "player"):
            for func in func_to_aggregate:
                for s in shifts:
                    data_series[f"{it}->{s}_{func.__name__}"].append(
                                                                func(
                                                                    make_diff_shifts(
                                                                        sootv[it],
                                                                        n=s
                                                                        )
                                                                    )
                                                                )
    return pd.DataFrame(data_series).set_index("id")


if __name__ == "__main__":
    base_dir = ".."
    input_dir = open( os.path.join(base_dir, "datadir.txt"), "r").read()[:-1]
    pred_dir = os.path.join(base_dir, "predictions")
    processed_dir = os.path.join(input_dir, "processed")


    func_to_aggregate = [
        np.max,
        scipy.stats.mode,
        np.mean,
        np.var,
        np.std,
        np.sum,
        scipy.stats.skew,
        linreg_trend,
    ]

    shifts = [
        0,1,3
    ]

    print("Reading")

    train_ser = pd.read_csv( os.path.join(input_dir,"processed", 'train_all_JSON.csv'), index_col=0 )["series"]
    test_ser = pd.read_csv( os.path.join(input_dir, "processed", 'test_all_JSON.csv'), index_col=0 )["series"]

    print("making train")
    new_data_train = get_data(train_ser, shifts, func_to_aggregate)

    print("making test")
    new_data_test = get_data(test_ser, shifts, func_to_aggregate)

    print("saving")

    new_data_train.to_csv( os.path.join(input_dir,"processed", 'train_timeseries.csv') )

    new_data_test.to_csv( os.path.join(input_dir,"processed", 'test_timeseries.csv') )

    print("Ezzz")
