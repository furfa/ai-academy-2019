from datetime import datetime
import os

def submit(pred, base_name="", pred_path="/home/furfa/work/ai-academy2019/predictions"):
    date = str(datetime.now())
    name = f"{base_name}[{date}].csv"
    path = os.path.join(pred_path, name)
    pred.to_csv(path, index = None) # 40 баллов
    print("File saved in :",path)