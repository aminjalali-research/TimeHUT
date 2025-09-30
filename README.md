# TimeHUT 


---


#### Basic AMC Training
```bash
#  WORKS: Basic TimeHUT with AMC and temperature scheduling

python train_unified_comprehensive.py Chinatown test_run \
    --loader UCR --scenario amc_temp \
    --amc-instance 1.0 --amc-temporal 0.5 --epochs 50

```


#### Ablation Study 
```bash

python ../../enhanced_metrics/timehut_comprehensive_ablation_runner.py \
    --dataset Chinatown --enable-gpu-monitoring --epochs 100

```

#### Unified Pipeline
```bash
python -m unified.master_benchmark_pipeline --models TimeHUT --datasets Chinatown --batch-size 8 --force-epochs 10 --timeout 300

```
python train_unified_comprehensive.py Chinatown optimized_efficient \
    --loader UCR --scenario amc_temp \
    --amc-instance 10.0 --amc-temporal 7.53 --amc-margin 0.3 \
    --min-tau 0.05 --max-tau 0.76 --t-max 25 \
    --epochs 120 --batch-size 16 \
    --temp-method polynomial_decay --temp-power 2.5 --verbose

```

#### **GPU Monitoring Options**
```bash
# Enable GPU monitoring (requires pynvml)
python timehut_comprehensive_ablation_runner.py --dataset Chinatown --enable-gpu-monitoring

# Disable GPU monitoring
python timehut_comprehensive_ablation_runner.py --dataset Chinatown

# Disable GFLOPs estimation
python timehut_comprehensive_ablation_runner.py --dataset Chinatown --disable-flops-estimation
```

### 2. Required Dependencies
```bash
# Install core dependencies (if needed)
pip install torch torchvision numpy scipy scikit-learn pandas matplotlib seaborn
pip install jupyter notebook tqdm psutil

# Optional optimization libraries
pip install optuna pyhopper neptune-client
```

#### Individual AMC Configuration Tests
```bash
# Test baseline (no AMC)
/home/anaconda3/envs/tslib/bin/python train_with_amc.py Chinatown baseline \
    --loader UCR --batch-size 8 --iters 50 \
    --amc-instance 0.0 --amc-temporal 0.0 --amc-margin 0.5 \
    --seed 42 --eval --dataroot /home/TSlib/datasets

# Test instance AMC only
/home/anaconda3/envs/tslib/bin/python train_with_amc.py Chinatown instance_only \
    --loader UCR --batch-size 8 --iters 50 \
    --amc-instance 1.0 --amc-temporal 0.0 --amc-margin 0.5 \
    --seed 42 --eval --dataroot /home/TSlib/datasets

# Test temporal AMC only  
/home/anaconda3/envs/tslib/bin/python train_with_amc.py Chinatown temporal_only \
    --loader UCR --batch-size 8 --iters 50 \
    --amc-instance 0.0 --amc-temporal 1.0 --amc-margin 0.5 \
    --seed 42 --eval --dataroot /home/TSlib/datasets

# Test both AMC components
/home/anaconda3/envs/tslib/bin/python train_with_amc.py Chinatown both_amc \
    --loader UCR --batch-size 8 --iters 50 \
    --amc-instance 1.0 --amc-temporal 1.0 --amc-margin 0.5 \
    --seed 42 --eval --dataroot /home/TSlib/datasets
```


```

### 🔍 Debug and Investigation Commands
```bash
# Verify unified script functionality

/home//anaconda3/envs/tslib/bin/python train_unified_comprehensive.py --help


# Test basic training with verbose output
/home/anaconda3/envs/tslib/bin/python train_unified_comprehensive.py Chinatown debug_run \
    --loader UCR --scenario amc_temp \
    --amc-instance 1.0 --amc-temporal 0.5 --epochs 5 --verbose
```

#### **Basic Scheduler Comparison**
Use the same hyperparameters for fair comparison:
```bash


# Base configuration (Chinatown optimized parameters):
BASE_PARAMS="--loader UCR --scenario amc_temp --amc-instance 10.0 --amc-temporal 7.53 --amc-margin 0.3 --min-tau 0.05 --max-tau 0.76 --t-max 25 --batch-size 8 --epochs 100 --verbose"

# Test cosine annealing (baseline)
/home/anaconda3/envs/tslib/bin/python train_unified_comprehensive.py Chinatown scheduler_cosine_annealing $BASE_PARAMS --temp-method cosine_annealing

# Test linear decay
/home/anaconda3/envs/tslib/bin/python train_unified_comprehensive.py Chinatown scheduler_linear_decay $BASE_PARAMS --temp-method linear_decay

# Test exponential decay
/home/anaconda3/envs/tslib/bin/python train_unified_comprehensive.py Chinatown scheduler_exponential_decay $BASE_PARAMS --temp-method exponential_decay --temp-decay-rate 0.95

# Test step decay
/home/anaconda3/envs/tslib/bin/python train_unified_comprehensive.py Chinatown scheduler_step_decay $BASE_PARAMS --temp-method step_decay --temp-step-size 8 --temp-gamma 0.5

# Test polynomial decay
/home/anaconda3/envs/tslib/bin/python train_unified_comprehensive.py Chinatown scheduler_polynomial_decay $BASE_PARAMS --temp-method polynomial_decay --temp-power 2.0

# Test sigmoid decay
/home/anaconda3/envs/tslib/bin/python train_unified_comprehensive.py Chinatown scheduler_sigmoid_decay $BASE_PARAMS --temp-method sigmoid_decay --temp-steepness 1.0

# Test constant temperature
/home/anaconda3/envs/tslib/bin/python train_unified_comprehensive.py Chinatown scheduler_constant $BASE_PARAMS --temp-method constant
```
#### **Baseline Benchmark (Establish Performance Metrics)**
```bash
cd /home/TSlib/models/timehut

# Quick baseline test (50 epochs)
python timehut_efficiency_optimizer.py --baseline-only --epochs 50

# Full baseline benchmark (200 epochs)
python timehut_efficiency_optimizer.py --baseline-only --epochs 200
```

#### **Verification Test**
```bash
# Test the optimized configuration
python train_unified_comprehensive.py Chinatown efficiency_verification \
    --loader UCR --scenario amc_temp --seed 2002 \
    --amc-instance 2.0 --amc-temporal 2.0 --amc-margin 0.5 \
    --min-tau 0.15 --max-tau 0.95 --t-max 25.0 \
    --batch-size 16 --epochs 120 \
    --temp-method polynomial_decay --temp-power 2.5

```

####  Basic TimeHUT Test 
```bash
python -m unified.master_benchmark_pipeline --models TimeHUT --datasets Chinatown --batch-size 8 --force-epochs 10 --timeout 300
```

####  Direct TimeHUT Training 
```bash
python train_with_amc.py Chinatown test_run \
    --loader UCR --batch-size 8 --iters 50 \
    --amc-instance 0.5 --amc-temporal 0.5 --amc-margin 0.5 \
    --seed 42 --eval --dataroot /home/datasets
```

# TimeHUT Model 


### Basic TimeHUT Training with AMC
```bash
# Basic TimeHUT with AMC and temperature scheduling
python train_unified_comprehensive.py dataset_name test_run \
    --loader UCR --scenario amc_temp \
    --amc-instance 1.0 --amc-temporal 0.5 --epochs 200
```

```bash
python train_unified_comprehensive.py dataset_name optimized_efficient \
    --loader UCR --scenario amc_temp \
    --amc-instance 10.0 --amc-temporal 7.53 --amc-margin 0.3 \
    --min-tau 0.05 --max-tau 0.76 --t-max 25 \
    --epochs 200 --batch-size 8 \
    --temp-method polynomial_decay --temp-power 2.5 --verbose
```

### Core Dependencies
```bash
pip install torch torchvision numpy scipy scikit-learn pandas matplotlib seaborn
pip install jupyter notebook tqdm psutil
```

### Ablation Studies
```bash
# Comprehensive ablation study with GPU monitoring
python ../../enhanced_metrics/timehut_comprehensive_ablation_runner.py \
    --dataset dataname --enable-gpu-monitoring --epochs 200
```

### Unified Pipeline
```bash
python -m unified.master_benchmark_pipeline \
    --models TimeHUT --datasets dataname \
    --batch-size 8 --force-epochs 200 
```

### Individual AMC Configuration Tests
```bash
# Test baseline (no AMC)
python train_with_amc.py dataname baseline \
    --loader UCR --batch-size 8 --iters 50 \
    --amc-instance 0.0 --amc-temporal 0.0 --amc-margin 0.5 \
    --seed 42 --eval

# Test instance AMC only
python train_with_amc.py dataname instance_only \
    --loader UCR --batch-size 8 --iters 50 \
    --amc-instance 1.0 --amc-temporal 0.0 --amc-margin 0.5 \
    --seed 42 --eval

# Test temporal AMC only  
python train_with_amc.py dataname temporal_only \
    --loader UCR --batch-size 8 --iters 50 \
    --amc-instance 0.0 --amc-temporal 1.0 --amc-margin 0.5 \
    --seed 42 --eval

# Test both AMC components
python train_with_amc.py dataname both_amc \
    --loader UCR --batch-size 8 --iters 50 \
    --amc-instance 1.0 --amc-temporal 1.0 --amc-margin 0.5 \
    --seed 42 --eval
```

##  Temperature Scheduler Comparison

Use the same hyperparameters for fair comparison:

```bash
# Base configuration (dataname optimized parameters)
BASE_PARAMS="--loader UCR --scenario amc_temp --amc-instance 10.0 --amc-temporal 7.53 --amc-margin 0.3 --min-tau 0.05 --max-tau 0.76 --t-max 25 --batch-size 8 --epochs 100 --verbose"

# Test different schedulers
python train_unified_comprehensive.py dataname scheduler_cosine_annealing $BASE_PARAMS --temp-method cosine_annealing
python train_unified_comprehensive.py dataname scheduler_linear_decay $BASE_PARAMS --temp-method linear_decay
python train_unified_comprehensive.py dataname scheduler_exponential_decay $BASE_PARAMS --temp-method exponential_decay --temp-decay-rate 0.95
python train_unified_comprehensive.py dataname scheduler_polynomial_decay $BASE_PARAMS --temp-method polynomial_decay --temp-power 2.0
```

##  GPU Monitoring Options

```bash
# Enable GPU monitoring (requires pynvml)
python timehut_comprehensive_ablation_runner.py --dataset dataname --enable-gpu-monitoring

# Disable GPU monitoring
python timehut_comprehensive_ablation_runner.py --dataset dataname

# Disable GFLOPs estimation
python timehut_comprehensive_ablation_runner.py --dataset dataname --disable-flops-estimation
```

##  Debug and Investigation

```bash
# Verify unified script functionality
python train_unified_comprehensive.py --help

# Test basic training with verbose output
python train_unified_comprehensive.py dataname debug_run \
    --loader UCR --scenario amc_temp \
    --amc-instance 1.0 --amc-temporal 0.5 --epochs 5 --verbose
```

##  Baseline Benchmark

```bash
# Quick baseline test (50 epochs)
python timehut_efficiency_optimizer.py --baseline-only --epochs 50

# Full baseline benchmark (200 epochs)
python timehut_efficiency_optimizer.py --baseline-only --epochs 200
```

```bash
# Test the optimized configuration
python train_unified_comprehensive.py dataname efficiency_verification \
    --loader UCR --scenario amc_temp --seed 2002 \
    --amc-instance 2.0 --amc-temporal 2.0 --amc-margin 0.5 \
    --min-tau 0.15 --max-tau 0.95 --t-max 25.0 \
    --batch-size 8 --epochs 120 \
    --temp-method polynomial_decay --temp-power 2.5
```









