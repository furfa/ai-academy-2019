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

base_dir = ".."
input_dir = open( os.path.join(base_dir, "datadir.txt"), "r").read()[:-1]
pred_dir = os.path.join(base_dir, "predictions")

print("input dir is ", input_dir)

def process_line(record):
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

# Processing
print("Processing train...")
train = process_file(os.path.join(input_dir, "dota2_skill_train.jsonlines"))
print("Processing test...")
test = process_file(os.path.join(input_dir, "dota2_skill_test.jsonlines"))

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
