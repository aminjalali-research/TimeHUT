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
```
TimeHUT (AtrialFibrillation)
train_unified_comprehensive.py AtrialFibrillation optimized_params --loader UEA --scenario amc_temp --amc-instance 2.04 --amc-temporal 0.08 --amc-margin 0.67 --min-tau 0.26 --max-tau 0.68 --t-max 49 --epochs 200 --verbose (18.79s, .4667)

train_unified_comprehensive.py Chinatown scheduler_exponential --loader UCR --scenario amc_temp --amc-instance 10.0 --amc-temporal 7.53 --amc-margin 0.3 --min-tau 0.05 --max-tau 0.76 --t-max 25 --temp-method exponential --batch-size 8 --epochs 200 --verbose

For TimeHUT ablation:
python timehut_comprehensive_ablation_runner.py --dataset Chinatown --enable-gpu-monitoring
/home/amin/anaconda3/envs/tslib/bin/python timehut_comprehensive_ablation_runner.py --dataset AtrialFibrillation --enable-gpu-monitoring

python compute_enhanced_timehut_ablation_runner.py --dataset Chinatown --enable-gpu-profiling --enable-flops-counting   (output:efficiency_summary_Chinatown_20250828_201625.csv)

Running all models: 
source activate tslib && python enhanced_metrics/all_models_runner.py --models TimeHUT_Top1,TimeHUT_Top2,TimeHUT_Top3,TS2vec,TimesURL,SoftCLT,CoST,CPC,TFC,TS_TCC,TLoss,TNC,MF_CLR --datasets Chinatown --timeout 300

conda activate tslib && python enhanced_metrics/enhanced_batch_runner.py --models BIOT,Ti_MAE,SimMTM,TFC,TimeHUT,VQ_MTM,MF_CLR,DCRNN,TS2vec,CoST,TS_TCC,TLoss,TimesURL,TNC --datasets AtrialFibrillation --timeout 200
```

Our model:
python timehut_comprehensive_ablation_runner.py --scenarios "AMC_Temperature_Cosine_AlgoOptim" "AMC_Temperature_MultiCycleCosine_AlgoOptim" "AMC_Temperature_MomentumAdaptive_AlgoOptim" --epochs 200

PyHopper Strategy for TimeHUT:
python enhanced_metrics/timehut_comprehensive_ablation_runner.py  --dataset AtrialFibrillation --enable-gpu-monitoring  --scenario AMC_PyHopper_BEST_46_67  --epochs 300


Dataset
         TimeHUT TS2Vec   TNC   TS-TCC  T-Loss   TST  TF-C
AF       0.53   0.200    0.133   0.267  0.200   0.067  0.200

Use the efficient version of TimeHUT, less flops and gpu memory, use optimized configuration for both AMC, then from that initial config optimize all param including each scheduler params



Practice talking frameworks: 1 thing, 2 types, 3 steps
They consider you as trainer!!! Damn frustrating!

run  each model and part in a separated file and put a proper name for it, use "edit" mode not "agent". solve one problem at the time. Never "agent" mode.

- no emoji and give root address instead of using my name
- prepare a chechlist and alway follow it
- Create doc folder (all documents), .github (all the initializations), separate files for documentation (put correct names for them and when you want to reuse does not occupy your window context)
how to code that does not destroy the other parts?
Docs folder: Bug_tracking.md, Implementation.md, Project_structure.md, UI_UX_doc.md













