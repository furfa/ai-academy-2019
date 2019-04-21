from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
sys.path.append("..")
import os
pd.set_option("max_columns", 10000)

import warnings
warnings.simplefilter("ignore")

import json

from tqdm import tqdm

json_in_df_name_flag = True
json_in_df_names = set()

def process_line(record):
    global json_in_df_name_flag
    for k in record.keys():
        item = record[k]
        if type( item ) == dict or type( item ) == list:
            record[k] = json.dumps(item)

            if json_in_df_name_flag:
                json_in_df_names.add(k)
    json_in_df_name_flag = False
    return record

def process_file(f_name):
    with open(f_name) as file:
        answer = list()
        for line in tqdm(file):
            rec= json.loads(line)
            # обработка record
            answer.append(
                process_line(rec)
            )
    return pd.DataFrame(answer).set_index("id")

if __name__ == "__main__":
    base_dir = ".."
    input_dir = open( os.path.join(base_dir, "datadir.txt"), "r").read()[:-1]
    pred_dir = os.path.join(base_dir, "predictions")

    print("input dir is ", input_dir)

    # Processing
    print("Processing train...")
    train = process_file(os.path.join(input_dir, "dota2_skill_train.jsonlines"))
    print("Processing test...")
    test = process_file(os.path.join(input_dir, "dota2_skill_test.jsonlines"))

    print("This cols are json: ",json_in_df_names)

    print("Saving")

    try:
        processed_dir = os.path.join(input_dir, "processed")
        os.mkdir(processed_dir)
    except:
        pass
    finally:
        train.to_csv(
            os.path.join(processed_dir, "train_all_JSON.csv")
        )
        test.to_csv(
            os.path.join(processed_dir, "test_all_JSON.csv")
        )
