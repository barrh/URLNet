from collections import defaultdict
import os

import pandas as pd
from pyarrow.parquet import ParquetFile
import pyarrow as pa

def read_parquet(dir_of_parquets:"str", max_data=0,split=[])->defaultdict:
    final_res = defaultdict(pd.DataFrame)
    data_splits = split
    for folder in os.listdir(dir_of_parquets):
        cur_folder = os.path.join(dir_of_parquets,folder)
        if not os.path.isdir(cur_folder):
            continue
        for file in os.listdir(cur_folder):
            if not file.endswith("parquet"):
                continue
            for split in data_splits:
                if split in folder:
                    if max_data==0:
                        ds = pd.read_parquet(os.path.join(dir_of_parquets,folder,file),columns=["normalized_url","label"])
                    elif len(final_res[split])<max_data:
                        pf = ParquetFile(os.path.join(dir_of_parquets, folder, file))
                        first_ten_rows = next(pf.iter_batches(batch_size=max_data))
                        ds = pa.Table.from_batches([first_ten_rows]).to_pandas()
                    final_res[split] = pd.concat([final_res[split],ds])
                    break
    return final_res
