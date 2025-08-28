```
https://github.com/DL4mHealth/Medformer (APAVA, TDBrain, ADFTD, PTB, PTB-XL datasets, 10 models Transformers)
https://github.com/chenguolin/NuTime  (UCR, UEA, SleepEDF, Epilepsy, etc. NuTime/src/data/preprocess.py
https://github.com/HaokunGUI/VQ_MTM  (VQ_MTM/models/    VQ_MTM/data_provider/data_factory.py   TUSZ dataset)
https://github.com/junwoopark92/Self-Supervised-Contrastive-Forecsating  ETTh1|ETTh2|ETTm1|ETTm2|Electricity|Traffic|Weather|Excange|Illness 10 models)
https://github.com/BorealisAI/ssl-for-timeseries (128 UCR, 30 UEA, 3 ETT datasets, Electricity, KPI dataset, knows how to submit jobs to slurm)
https://github.com/findalexli/TimeseriesContrastiveModels (CLOCS Mixing-up SimCLR TS-TCC	TS2Vec TFC   SleepEEG/Epilepsy/FD-A//FD-B/HAR/Gesture/ECG/EMG)
https://github.com/DL4mHealth/SLOTS (SLOTS/Mixing-up/SimCLR/TS-TCC/TS2Vec/TFC       DEAP/SEED/EPILEPSY/HAR/P19)  
https://github.com/DL4mHealth/LEAD/ (GREAT)

https://github.com/lanxiang1017/DynamicBadPairMining_ICLR24
https://github.com/blacksnail789521/TimeDRL
https://github.com/LiuZH-19/CTRL (128 UCR, 30 UEA, 3 ETT datasets, Exchange, Wind, ILI, Weather)
https://github.com/yurui12138/TS-DRC
https://github.com/theabrusch/Multiview_TS_SSL
https://github.com/maxxu05/relcon (The Opportunity, PAMAP2, HHAR, and Motionsense datasets 

https://github.com/TsingZ0/FedTGP
https://github.com/duyngtr16061999/KDMCSE
https://github.com/sfi-norwai/eMargin
https://github.com/yingxiatang/FreConvNet
```

## Failure Cases and Limitations of timehut
metrics and computation factors

TS2Vec:
train_unified_comprehensive.py Chinatown no_scheduling_test --loader UCR --scenario baseline --epochs 200 --temp-method no_scheduling --verbose (7.05s, .9796)

TS2Vec+Scheduler
train_unified_comprehensive.py Chinatown optimized_params --loader UCR --scenario baseline --epochs 200 --verbose (7.35s. .9796)

TimeHUT
train_unified_comprehensive.py Chinatown optimized_params --loader UCR --scenario amc_temp --amc-instance 10.0 --amc-temporal 7.53 --amc-margin 0.3 --min-tau 0.05 --max-tau 0.76 --t-max 25 --epochs 200 --verbose (10.90s, .9825)

TimeHUT (AtrialFibrillation)
train_unified_comprehensive.py AtrialFibrillation optimized_params --loader UEA --scenario amc_temp --amc-instance 2.04 --amc-temporal 0.08 --amc-margin 0.67 --min-tau 0.26 --max-tau 0.68 --t-max 49 --epochs 200 --verbose (18.79s, .4667)

train_unified_comprehensive.py Chinatown scheduler_exponential --loader UCR --scenario amc_temp --amc-instance 10.0 --amc-temporal 7.53 --amc-margin 0.3 --min-tau 0.05 --max-tau 0.76 --t-max 25 --temp-method exponential --batch-size 8 --epochs 200 --verbose


timehut_efficiency_optimizer.py --full-optimization --dataset Chinatown --epochs 200

python train_unified_comprehensive.py Chinatown optimized_efficient --loader UCR --scenario amc_temp --amc-instance 10.0 --amc-temporal 7.53 --amc-margin 0.3 --min-tau 0.05 --max-tau 0.76 --t-max 25 --epochs 120 --batch-size 16 --temp-method polynomial_decay --temp-power 2.5 --verbose




