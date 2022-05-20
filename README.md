# AICUP - Orchid Classifier

![img]()

# Experiments

Model Definition & Training Epoch

| Model \ Detail            | Pretrained                         | Epoch | Training Time (T4) |
| ------------------------- | ---------------------------------- | ----- | ------------------ |
| 1. ResNet50               | yes                                | 20    |                    |
| 2. ViT                    | google/vit-base-patch16-224-in21k  | 10    | 9 mins             |
| 3. ViT                    | gary109/vit-base-orchid219-demo-v5 | 15    | 14 mins            |
| 4. Self-defined CNN       |                                    | 50    | 53 mins            |
| 5. ViT with Augumentation | gary109/vit-base-orchid219-demo-v5 | 15    | 15 mins            |
| 5. ViT with STN           | gary109/vit-base-orchid219-demo-v5 | 50    |                    |

Model performance :

| Model \ Performance on dataset | Train Set (80%) | Valid Set (20%) |
| ------------------------------ | --------------- | --------------- |
| Model 1                        | 96.0 %          | 58.0 %          |
| Model 2                        | 95.6 %          | 82.6 %          |
| Model 3 (in epoch 10)          | 100 %           | 90.0 %          |
| Model 3 (in epoch 15)          | 100 %           | 92.9 %          |
| Model 4                        | 84.3%           | 67.7%           |
| Model 5 (in epoch 10)          | 95.8 %          | 84.2 %          |
| Model 5 (in epoch 15)          | 95.8 %          | 87.7 %          |
| Model 5 (in epoch 10)          | 100 %           | 100 %           |
| Model 5 (in epoch 50)          | 100 %           | 100 %           |
|                                |                 |                 |

# Installation

```bash
    git clone https://github.com/jimmylin0979/AICUP2022-OrchidClassifier.git
    cd AICUP2022-OrchidClassifier
    pip install -r requirements.txt
```

# Getting Started

Configuration files of models used for experiments are located inside ./config.py file. You may edit these files depending upon the location of datasets, ratio of how to split train/valid set, .., and so on.

## Training

```python
    CUDA_VISIBLE_DEVICES=0 python main.py
```

Noted that default location of datasets in the config.py are `./data/dataset/train`.
Please make change accordingly.

## Testing

```python
    CUDA_VISIBLE_DEVICES=0 python evaluate.py
```

# References
