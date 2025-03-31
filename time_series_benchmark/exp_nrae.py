import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import random
from datetime import datetime
import os
import sys
sys.path.append("./src/")
from dataset import train_val_test_split_with_knn
from nrae_trainer import fit
from models.nrae import NRAE
from dicts import index, fields
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# get file date and time
now = datetime.now()
date = now.strftime("%Y-%m-%d")
time = now.strftime("%H-%M-%S")

# randomness
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# load config
with open("./configs/nrae.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# exp
m = cfg["m"]
variants = cfg["variants"]
history = cfg["history"]
horizons = cfg["horizons"]
batch_sizes = cfg["batch_sizes"]
lrs = cfg["lrs"]
seeds = cfg["seeds"]
epochs = cfg["epochs"]

variants_str = "-".join(str(item) for item in variants)
datasets_str = "-".join(str(item["name"]) for item in cfg["datasets"])
seeds_str = "-".join(str(item) for item in seeds)

# results
df_results = pd.DataFrame(columns=index+fields)
df_results = df_results.set_index(index)

if __name__ == "__main__":
    for batch_size in batch_sizes:
        for seed in seeds:
            for lr in lrs:
                for h_idx, horizon in enumerate(horizons):
                    for d in cfg["datasets"]:
                        # params
                        d_embedding = d["params"]["d_embedding"]
                        num_neighbors = cfg["params"]["num_neighbors"][0]
                        lambda_val = cfg["params"]["lambda"][0]
                        
                        # load data
                        data = np.load(d["file"])
                        if "take_n" in d:
                            data = data[:d["take_n"]]

                        # datasets
                        n_steps, n_channels = data.shape
                        f_train, f_val, f_test = d["split"]
                        split_str = "".join(str(int(item*10)) for item in d["split"])
                        
                        f_train, f_val, f_test = d["split"]
                        split_str = "".join(str(int(item*10)) for item in d["split"])

                        dataset_train, dataset_val, dataset_test, embedding = train_val_test_split_with_knn(
                            data, 
                            [f_train, f_val, f_test],
                            history, 
                            horizon, 
                            k_neighbors=num_neighbors,
                            d_embedding=d_embedding
                        )
                        
                        print(f"Batch size: {batch_size}, seed: {seed}, lr: {lr}, horizon: {horizon}")
                        print(f"{d['name']} ({len(data)}, {len(dataset_train)}, {len(dataset_val)}, {len(dataset_test)})")

                        for v in variants:
                            # random state
                            random.seed(seed)
                            np.random.seed(seed)
                            torch.manual_seed(seed)
                            torch.cuda.manual_seed_all(seed)

                            # dataloaders
                            dataloader_train = DataLoader(dataset_train, batch_size, shuffle=True, num_workers=4)
                            dataloader_val = DataLoader(dataset_val, batch_size, shuffle=True, num_workers=4)
                            dataloader_test = DataLoader(dataset_test, batch_size, shuffle=False, num_workers=4)
                            dataloaders = [dataloader_train, dataloader_val, dataloader_test]

                            # model/variant
                            kernel = {'type': 'binary', 'lambda': lambda_val}
                            approx_order = 1 if v == "ApproxOrd1" else 2

                            d_in = n_channels * history
                            d_out = n_channels * horizon
                            l_sz = 256

                            model = NRAE(
                                d_in=d_in,
                                d_out=d_out,
                                l_sz=l_sz,
                                approx_order=approx_order,
                                kernel=kernel
                            )


                            optimizer = Adam(model.parameters(), lr=lr)
                            loss_fn = nn.MSELoss()
                            metric_fn = nn.L1Loss()

                            # fit
                            _, history_model = fit(model, optimizer, loss_fn, metric_fn, epochs, *dataloaders, desc=m + "/" + v)

                            # log
                            run = []
                            for epoch_id, metrics in enumerate(history_model):
                                run.append([
                                    d["name"], n_steps, split_str, epochs, history, horizon,
                                    m, v, seed, batch_size, lr, num_neighbors, d_embedding,
                                    epoch_id,
                                    metrics[0,0], metrics[0,1], metrics[0,2],  # mse
                                    metrics[1,0], metrics[1,1], metrics[1,2],  # mae
                                ])

                            df_run = pd.DataFrame(run, columns=index+fields)
                            df_run = df_run.set_index(index)
                            df_results = pd.concat([df_results, df_run])

                            # save logs
                            df_results.to_parquet(f"./results/{date}_{time}_NRAE_{variants_str}_d_{datasets_str}_e_{epochs}_s_{seeds_str}.parquet")

                            # save checkpoint
                            PATH = f"./checkpoints/{date}_{time}_NRAE_{variants_str}_d_{datasets_str}_e_{epochs}_s_{seeds_str}/"
                            os.makedirs(PATH, exist_ok=True)
                            torch.save(model.state_dict(), PATH + f"{date}_{time}_NRAE_{v}_s-{seed}_{d['name']}_{n_steps}_{split_str}_e-{epochs}_L-{history}_H-{horizon}_bs-{batch_size}_lr-{lr}_nn-{num_neighbors}_emb-{d_embedding}.pt")
                        
                        print("\r")
