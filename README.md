# EvoPathGraph

This repository contains the code for the paper **"EvoPathGraph: An Evolvable Pathology Graph Benchmark Dataset for Multi-Domain and Multi-Task Evaluation"**.

## Running Environment

- Python 3.8.20
- torch 2.4.0+cu121
- dgl 2.0.0.cu121
- numpy 1.24.4
- pandas 2.0.3
- scikit-learn 1.3.2
- scikit-survival 0.22.2

## Datasets
Our dataset is avaliable at https://drive.google.com/drive/folders/1s2usKRcJuxYnDb-6baPUHbvdCkPQMoTh?usp=drive_link


## Experiments

### 1. Overall Survival Prediction
Predict overall survival time for patients.
```bash
python main1.py --model gcn --dataset_name blca --task OS
```

### 2. Progression-free Survival Prediction
Predict progression-free survival for patients.
```bash
python main1.py --model gcn --dataset_name blca --task PFS
```

### 3. Tumor Stage Prediction
Predict tumor stage classification for patients.
```bash
python main1.py --model gcn --dataset_name blca --task Stage
```

### 4. Tumor Type Classification
Multi-class cancer type classification across 5 different cancer types.
```bash
python main2_cancer_cls.py --model gcn --dataset_name all --task CancerCls --n_classes 5
```

### 5. Cross-domain Generalization Analysis

Evaluate model generalization across different domains:

#### Cross-center Generalization
Validate model performance across different medical centers.
```bash
python main3_cross_center.py --model gcn --dataset_name blca --task OS --domain BT
```

#### Cross-age Generalization
Test model robustness across different age groups.
```bash
python main4_cross_age.py --model gcn --dataset_name blca --task OS --domain 0
```

#### Cross-cancer Type Generalization
Evaluate model transferability across different cancer types.
```bash
python main5_cross_cancer.py --model gcn --dataset_name blca --task OS --domain stad
```