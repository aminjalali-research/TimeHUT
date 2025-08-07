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
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
labeled_ratio = 0.7
run_dir = 'SAMPLE'
def train_model(config, temp_dictionary = None , amc_setting = None, type = "full"):
    '''
    trains the ts2vec model using either full dataset or the split dataset
    '''

    
    if type == 'full':
        # print("Training the final model")
        t = time.time()
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
        if task_type == 'classification':
            out, eval_res = tasks.eval_classification(model, train_data, train_labels,test_data, test_labels, eval_protocol='svm')
        elif task_type == 'forecasting':
            out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols)
        elif task_type == 'anomaly_detection':
            out, eval_res = tasks.eval_anomaly_detection(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
        elif task_type == 'anomaly_detection_coldstart':
            out, eval_res = tasks.eval_anomaly_detection_coldstart(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
        else:
            assert False

        # if task_type != 'classification':

        #     # pkl_save(f'{run_dir}/out.pkl', out)
        #     # pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
        #     # print('Evaluation result:', eval_res)
        # else:
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
    
    out_dir = "results_forecast/ETT"
    os.makedirs(out_dir, exist_ok=True)


    
    exp = 0
    n = 0
    
    for repeat in range(1):
        for amc_instance in [0.1 ]:
            for amc_temporal in [0, 0.1, 0.5, 1, 3]:
                for tau_min in [0.07, 0.1, 0.2]:
                    for tau_max in [0.7, 0.8, 1.0]:
                            for t_max in [5, 10, 25]:
                                accs = []
                                auprcs = []
                                margin = 0.5
                            
                                t1 = time.time()
                                device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
                
                                print('Loading data... ', end='')
                                if args.loader == 'UCR':
                                    task_type = 'classification'
                                    train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset, root = args.dataroot)
 
                                elif args.loader == 'UEA':
                                    task_type = 'classification'
                                    train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset, root = args.dataroot)
                                elif args.loader == 'forecast_csv':
                                    task_type = 'forecasting'
                                    data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset)
                                    train_data = data[:, train_slice]
                                    
                                elif args.loader == 'forecast_csv_univar':
                                    task_type = 'forecasting'
                                    data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset, univar=True)
                                    train_data = data[:, train_slice]
                                    
                                elif args.loader == 'forecast_npy':
                                    task_type = 'forecasting'
                                    data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset)
                                    train_data = data[:, train_slice]
                                    
                                elif args.loader == 'forecast_npy_univar':
                                    task_type = 'forecasting'
                                    data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset, univar=True)
                                    train_data = data[:, train_slice]
                                    
                                elif args.loader == 'anomaly':
                                    task_type = 'anomaly_detection'
                                    all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
                                    train_data = datautils.gen_ano_train_data(all_train_data)
                                    
                                elif args.loader == 'anomaly_coldstart':
                                    task_type = 'anomaly_detection_coldstart'
                                    all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
                                    train_data, _, _, _ = datautils.load_UCR('FordA')
                                    
                                else:
                                    raise ValueError(f"Unknown loader {args.loader}.")
                                

                                config = dict(
                                    batch_size=args.batch_size,
                                    lr=args.lr,
                                    output_dims=args.repr_dims,
                                    max_train_length=args.max_train_length
                                )
                                temp_settings = None
                                temp_settings = {}
                                temp_settings['min_tau'] = tau_min
                                temp_settings['max_tau'] = tau_max
                                temp_settings['t_max'  ] = t_max

                                amc_settings = {}
                                amc_settings['amc_instance'] = amc_instance
                                amc_settings['amc_temporal'] = amc_temporal
                                amc_settings['amc_margin'] = margin
                                

                                final_res = train_model(config, temp_settings, amc_setting= amc_settings, type = "full")
            
                                run_name = "_".join((args.loader, args.dataset, "AMC", f"{n}.json"))
                                out_name = os.path.join(out_dir, run_name)

                                out = {}
                                
                                output = {}
                                output['temp setting'] = temp_settings
                                output['amc setting'] = amc_settings
                                output['res'] = final_res
                                # accs.append(final_res['acc'])
                                # auprcs.append(final_res['auprc'])
                                
                                # print('temp setting : ', temp_settings, 'amc setting : ', amc_settings, "\nfinal res : ", final_res)
                                print('amc setting : ', amc_settings, 'temp: ', temp_settings,   "\nfinal res : ", final_res)
                                # print(final_res)
                                
                                out[exp] = output
                                
                                if exp == 0:
                                    n = 0
                                    while os.path.isfile(out_name) is True:
                                        out_name = out_name.replace(f"{n}.json", f"{n+1}.json")
                                        n = n+1
                                        
                                    print(f'{out_name} created.')
                                    out_file = open(out_name, "w")
                                    json.dump(out, out_file)
                                    out_file.close()
                                else:
                                    # print("Loaded!")
                                    outjson =json.load(open(out_name))
                                    outjson.update(out)
                                    out_file = open(out_name, "w")
                                    json.dump(outjson, out_file)
                                    out_file.close()
                                exp = exp +1
                                t2 = time.time()
                                print('Elapsed Time : ', np.round(t2 - t1, 2))
                                print("Finished.")
                                import gc
                                torch.cuda.empty_cache()
                                gc.collect()
                            
    
    for repeat in range(1):
        for amc_instance in [0.1]:
                            for amc_temporal in [0, 0.1, 0.5, 1, 3]:

                                accs = []
                                auprcs = []
                                margin = 0.5
                            
                                t1 = time.time()
                                device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
                
                                print('Loading data... ', end='')
                                if args.loader == 'UCR':
                                    task_type = 'classification'
                                    train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset, root = args.dataroot)
 
                                elif args.loader == 'UEA':
                                    task_type = 'classification'
                                    train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset, root = args.dataroot)
                                elif args.loader == 'forecast_csv':
                                    task_type = 'forecasting'
                                    data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset)
                                    train_data = data[:, train_slice]
                                    
                                elif args.loader == 'forecast_csv_univar':
                                    task_type = 'forecasting'
                                    data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset, univar=True)
                                    train_data = data[:, train_slice]
                                    
                                elif args.loader == 'forecast_npy':
                                    task_type = 'forecasting'
                                    data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset)
                                    train_data = data[:, train_slice]
                                    
                                elif args.loader == 'forecast_npy_univar':
                                    task_type = 'forecasting'
                                    data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset, univar=True)
                                    train_data = data[:, train_slice]
                                    
                                elif args.loader == 'anomaly':
                                    task_type = 'anomaly_detection'
                                    all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
                                    train_data = datautils.gen_ano_train_data(all_train_data)
                                    
                                elif args.loader == 'anomaly_coldstart':
                                    task_type = 'anomaly_detection_coldstart'
                                    all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
                                    train_data, _, _, _ = datautils.load_UCR('FordA')
                                    
                                else:
                                    raise ValueError(f"Unknown loader {args.loader}.")
                                

                                config = dict(
                                    batch_size=args.batch_size,
                                    lr=args.lr,
                                    output_dims=args.repr_dims,
                                    max_train_length=args.max_train_length
                                )
                                temp_settings = None
                                # temp_settings = {}
                                # temp_settings['min_tau'] = tau_min
                                # temp_settings['max_tau'] = tau_max
                                # temp_settings['t_max'  ] = t_max

                                amc_settings = {}
                                amc_settings['amc_instance'] = amc_instance
                                amc_settings['amc_temporal'] = amc_temporal
                                amc_settings['amc_margin'] = margin
                                

                                final_res = train_model(config, temp_settings, amc_setting= amc_settings, type = "full")
            
                                run_name = "_".join((args.loader, args.dataset, "AMC", f"{n}.json"))
                                out_name = os.path.join(out_dir, run_name)

                                out = {}
                                
                                output = {}
                                output['temp setting'] = temp_settings
                                output['amc setting'] = amc_settings
                                output['res'] = final_res
                                # accs.append(final_res['acc'])
                                # auprcs.append(final_res['auprc'])
                                
                                # print('temp setting : ', temp_settings, 'amc setting : ', amc_settings, "\nfinal res : ", final_res)
                                print('amc setting : ', amc_settings, 'temp: ', temp_settings,   "\nfinal res : ", final_res)
                                # print(final_res)
                                
                                out[exp] = output
                                
                                if exp == 0:
                                    n = 0
                                    while os.path.isfile(out_name) is True:
                                        out_name = out_name.replace(f"{n}.json", f"{n+1}.json")
                                        n = n+1
                                        
                                    print(f'{out_name} created.')
                                    out_file = open(out_name, "w")
                                    json.dump(out, out_file)
                                    out_file.close()
                                else:
                                    # print("Loaded!")
                                    outjson =json.load(open(out_name))
                                    outjson.update(out)
                                    out_file = open(out_name, "w")
                                    json.dump(outjson, out_file)
                                    out_file.close()
                                exp = exp +1
                                t2 = time.time()
                                print('Elapsed Time : ', np.round(t2 - t1, 2))  
                                print("Finished.")
                                import gc
                                torch.cuda.empty_cache()
                                gc.collect()
