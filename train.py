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
import pyhopper
import json

labeled_ratio = 0.7

def train_model(config, temp_dictionary = None , type = "full"):
    '''
    trains the ts2vec model using either full dataset or the split dataset
    '''
    if type == 'split':
        # print("Training the split model")
        t = time.time()
        model = TS2Vec( # Trains ts2vefc in a ssl fashion
            input_dims=train_data.shape[-1],
            device=device,
            temp_dictionary = temp_dictionary,
            **config
        )
        loss_log = model.fit(
            train_data_split,
            n_epochs=args.epochs,
            n_iters=args.iters,
            verbose=False
        )
        t = time.time() - t
        # print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")
        out_val, eval_res_val = tasks.eval_classification(model, train_data_split, train_labels_split, val_data_split, val_labels_split, eval_protocol='svm')
        print('Evaluation result (val)               :', eval_res_val)
        out_test, eval_res_test = tasks.eval_classification(model, train_data_split, train_labels_split, test_data, test_labels, eval_protocol='svm')
        print('Evaluation result (test)              :', eval_res_test)
        return eval_res_val
    
    if type == 'full':
           # print("Training the final model")
        t = time.time()
        model = TS2Vec( # Trains ts2vefc in a ssl fashion
            input_dims=train_data.shape[-1],
            device=device,
            temp_dictionary = temp_dictionary,
            **config
        )
        loss_log = model.fit(
            train_data,
            n_epochs=args.epochs,
            n_iters=args.iters,
            verbose=False
        )

        out, eval_res = tasks.eval_classification(model, train_data, train_labels,test_data, test_labels, eval_protocol='svm')
        print('Evaluation result on test (full train):', eval_res)
        return eval_res
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
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
    args = parser.parse_args()
    
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))
    
    out_dir = "results/"
    os.makedirs(out_dir, exist_ok=True)
    run_name = "_".join((args.loader, args.dataset, args.method, "MC.json"))
    out_name = os.path.join(out_dir, run_name)
    
    
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    
    print('Loading data... ', end='')
    if args.loader == 'UCR':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset, root = args.dataroot)
        train_data_split, val_data_split, train_labels_split, val_labels_split = train_test_split(train_data, train_labels, 
                                                        test_size=labeled_ratio, random_state=101)
        
    elif args.loader == 'UEA':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset, root = args.dataroot)
        train_data_split, val_data_split, train_labels_split, val_labels_split = train_test_split(train_data, train_labels, 
                                                        test_size=labeled_ratio, random_state=101)
    
    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length
    )
    

    
    # define hopper objective
    def objective(hparams: dict):
        temp_settings = {}
        temp_settings['min_tau'] = hparams['min_tau']
        temp_settings['max_tau'] = hparams['max_tau']
        temp_settings['t_max'] = hparams['t_max']
        out = train_model(config, temp_settings, type = "split")
        return out['acc'] #+ out['auprc']
        
    search = pyhopper.Search(
    {
        "min_tau": pyhopper.float(0,0.3, "0.05f"),
        "max_tau": pyhopper.float(0.5,1, "0.05f"),
        "t_max"  : pyhopper.float(1,20, "0.5f"),
        # "t_max"  : pyhopper.float(1e1,5,"0.1g")
    }
    )
    # temp_settings = search.run(objective, "maximize", "20m", n_jobs=1)
    temp_settings = search.run(objective, "maximize", steps=10, n_jobs=1)
    print("Best Params : ", temp_settings)
    
    # Training the split model
    print("Training the split model")
    
    # train_model(config, temp_settings, type = "split")
    final_res = train_model(config, temp_settings, type = "full")

    output = {}
    output['best'] = temp_settings
    output['res'] = final_res
    out_file = open(out_name, "w")
    json.dump(output, out_file)
    
    print("Finished.")
