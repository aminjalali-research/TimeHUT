import pandas as pd
import numpy as np
import pandas as pd
amc_best = pd.read_csv('/media/milad/DATA/TSResearch/SpaCon/ts2vec/AMC_complete_results.csv')
amc_best = amc_best[['Dataset', 'accBase', 'accAMC', 'Ci/Ct (acc)', 'auprcAMC']]

temp_best = pd.read_csv('temp_results_compare.csv')

merged_df = temp_best[['Dataset', 'AccBase']].copy()
merged_df['ci'] = amc_best['Ci/Ct (acc)'].apply(lambda x:x.split('/')[0]).values
merged_df['ct'] = amc_best['Ci/Ct (acc)'].apply(lambda x:x.split('/')[1]).values
merged_df['min_tau'] = temp_best['min_tau / max_tau / t_max'].apply(lambda x:x.split('/')[0]).values
merged_df['max_tau'] = temp_best['min_tau / max_tau / t_max'].apply(lambda x:x.split('/')[1]).values
merged_df['t_max'] = temp_best['min_tau / max_tau / t_max'].apply(lambda x:x.split('/')[2]).values

merged_df['temp_acc'] = temp_best['Acc'].values
merged_df['temp_auprc'] = temp_best['Auprc'].values
merged_df['amc_acc'] = amc_best['accAMC']
merged_df['amc_auprc'] = amc_best['auprcAMC']


import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
from ts2vec import TS2Vec
import tasks
import datautils
from utils import init_dl_program
from sklearn.model_selection import train_test_split
import json

parser = argparse.ArgumentParser()
# parser.add_argument('dataset', help='The dataset name')
# parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
# parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
parser.add_argument('--seed', type=int, default=None, help='The random seed')
parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
parser.add_argument('--method', type=str, default='acc', help='whether to choose acc or auprc or both')
parser.add_argument('--dataroot', type=str, default='/media/milad/DATA/TSResearch/datasets', help='root for the dataset')
args = parser.parse_args('')
args.loader = 'UCR'
# print("Dataset:", args.dataset)
# print("Arguments:", str(args))

def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])

labeled_ratio = 0.7

def train(dataset, args,  temp_dictionary = None, amc_setting = None):


    args.dataset = dataset
    out_dir = "results/results_gs"
    os.makedirs(out_dir, exist_ok=True)

    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

    # print('Loading data... ', end='')
    if args.loader == 'UCR':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset, root = args.dataroot)

    elif args.loader == 'UEA':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset, root = args.dataroot)


    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length
    )


    model = TS2Vec( # Trains ts2vefc in a ssl fashion
    input_dims=train_data.shape[-1],
    device=device,
    temp_dictionary = temp_dictionary,
    amc_setting =amc_setting,
    **config

    )
    loss_log = model.fit(
        train_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=False
    )

    out, eval_res = tasks.eval_classification(model, train_data, train_labels,test_data, test_labels, eval_protocol='svm')
    return eval_res


np.seterr(all="ignore")
import warnings
warnings.filterwarnings('ignore')
for r in range(len(merged_df)):
    row = merged_df.loc[r]
    temp_setting = row[['min_tau', 'max_tau', 't_max']].astype(float).to_dict()
    amc_setting = {}
    amc_setting['amc_instance'] = float(row['ci'])
    amc_setting['amc_temporal'] = float(row['ct'])
    print(row.Dataset)
    print('only temp : ', row['temp_acc'], row['temp_auprc'])
    print('only amc : ', row['amc_acc'], row['amc_auprc'])
    print('using both')
    for i in range(3):
        print(train(row.Dataset, args, temp_dictionary=temp_setting, amc_setting=amc_setting))
    print('###################################')
row