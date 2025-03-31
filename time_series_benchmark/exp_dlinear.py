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
from dataset import train_val_test_split
from trainer import fit

from models.dlinear import DLinear, DLinearHyper

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
with open("./configs/dlinear.yaml", "r") as f:
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
    # for each batch size
    for batch_size in batch_sizes:

        # for each seed
        for seed in seeds:

            # for each learning rate
            for lr in lrs:

                # for each horizon
                for h_idx, horizon in enumerate(horizons):

                    # for each dataset
                    for d in cfg["datasets"]:
                        
                        # params
                        d_embedding = d["params"]["d_embedding"]
                        if len(horizons) == len(cfg["params"]["d_hyper_hidden"]):
                            d_hyper_hidden = cfg["params"]["d_hyper_hidden"][h_idx]
                        else:
                            d_hyper_hidden = cfg["params"]["d_hyper_hidden"][0] # default
                        epochs = cfg["epochs"]
                        if "epochs" in d:
                            epochs = d["epochs"]
                        
                        # load data
                        data = np.load(d["file"])
                        if "take_n" in d:
                            data = data[:d["take_n"]]

                        # datasets
                        n_steps, n_channels = data.shape
                        f_train, f_val_, f_test = d["split"]
                        split_str = "".join(str(int(item*10)) for item in d["split"])
                        dataset_train, dataset_val, dataset_test, embedding = train_val_test_split(data, [f_train, f_test], history, horizon, d_embedding=d_embedding)
                        
                        print("Batch size: {}, seed: {}, lr: {}, horizon: {}".format(batch_size, seed, lr, horizon))
                        print("{} ({}, {}, {}, {}), H={}, E={}\r\n-----".format(d["name"], len(data), len(dataset_train), len(dataset_val), len(dataset_test), d_hyper_hidden, d_embedding))

                        # for each variant
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
                            if v=="ApproxOrd1":
                                model = DLinear(history, horizon, n_channels)
                            elif v=="ApproxOrd1":
                                model = DLinearHyper(history, horizon, d_hyper_hidden, n_input=n_channels, d_embedding=d_embedding)

                            optimizer = Adam(model.parameters(), lr=lr)
                            loss_fn = nn.MSELoss()
                            metric_fn = nn.L1Loss()

                            # fit
                            _, history_model = fit(model, optimizer, loss_fn, metric_fn, epochs, *dataloaders, desc=m + "/" + v)

                            # log
                            run = []
                            for epoch_id, metrics in enumerate(history_model):
                                run.append([
                                    d["name"],      # dataset
                                    n_steps,        # size
                                    split_str,      # split
                                    epochs,         # epochs
                                    history,        # history
                                    horizon,        # horizon
                                    m,              # model
                                    v,              # variant
                                    seed,           # seed
                                    batch_size,     # batch_size
                                    lr,             # lr
                                    d_hyper_hidden, # d_hyper_hidden
                                    d_embedding,    # d_embedding        
                                    
                                    epoch_id,       # epoch id
                                    metrics[0,0],   # mse_train
                                    metrics[0,1],   # mse_val
                                    metrics[0,2],   # mse_test
                                    metrics[1,0],   # mae_train
                                    metrics[1,1],   # mae_val
                                    metrics[1,2],   # mae_test
                                ])

                            df_run = pd.DataFrame(run, columns=index+fields)
                            df_run = df_run.set_index(index)
                            df_results = pd.concat([df_results, df_run])

                            # save logs
                            df_results.to_parquet("./results/{}_{}_DLinear_{}_d_{}_e_{}_s_{}.parquet".format(date, time, variants_str, datasets_str, epochs, seeds_str))

                            # save checkpoint
                            PATH = "./checkpoints/{}_{}_DLinear_{}_d_{}_e_{}_s_{}/".format(date, time, variants_str, datasets_str, epochs, seeds_str)
                            os.makedirs(PATH, exist_ok=True)
                            torch.save(model.state_dict(), PATH + "{}-{}_DLinear_{}_s-{}_{}_{}_{}_e-{}_L-{}_H-{}_bs-{}_lr-{}_hh-{}_emb-{}.pt".format(
                                date,
                                time,
                                v,
                                seed,
                                d["name"],
                                n_steps,
                                split_str,
                                epochs,
                                history,
                                horizon,
                                batch_size,
                                lr,
                                d_hyper_hidden,
                                d_embedding,
                            ))
                        
                        print("\r")
