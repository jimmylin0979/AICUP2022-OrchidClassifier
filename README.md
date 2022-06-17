# AICUP - Orchid Classifier

# Model Zoo

|            name            | pretrain | resolution | acc | #params |                                   links-model                                    |                                    links-ema                                     |
| :------------------------: | :------: | :--------: | :-: | :-----: | :------------------------------------------------------------------------------: | :------------------------------------------------------------------------------: |
|     model_stnvit_mixs      |          |    384     |  x  |         | https://drive.google.com/uc?export=download&id=16RLc7MQmCXuRn_PuN31Js5OsgRIECRBH | https://drive.google.com/uc?export=download&id=10rmlJGUQ5LGkRppXQj1qevFmFtiuk3EU |
|         model_swin         |          |    384     |  x  |         | https://drive.google.com/uc?export=download&id=1R5SSLjuUGA7NdCNVkzMinO59Vo9U38lh | https://drive.google.com/uc?export=download&id=1tqchfh4MpLnI57ZaqJDQKmG2-qT9HcP5 |
|  model_swin_mixs_tmax101   |          |    384     |  x  |         | https://drive.google.com/uc?export=download&id=1yhbiOHs4KH4frMLd3vdYADpjvNemJ378 | https://drive.google.com/uc?export=download&id=1_h60KHU3yFRZjectA2GAP7r7tV-znSSg |
| model_convnext_384_tmax101 |          |    384     |  x  |         | https://drive.google.com/uc?export=download&id=1mzRVoejVL_pnqjN28XcKLNnqqnTNVdUz | https://drive.google.com/uc?export=download&id=1II7Vd6CZqyXNywSSICXFmMdYwQAAIkkZ |

If you want to used the pretrained model to evaluate the result directly, please save the model.pth, ema.pth into `./saved/<model_name>/model.pth`, which is the default path for evaluate.py to get model preatrined werights.

# Installation

```bash
    git clone https://github.com/jimmylin0979/AICUP2022-OrchidClassifier.git
    cd AICUP2022-OrchidClassifier
    pip install -r requirements.txt
```

# Getting Started

Configuration files of models used for experiments are located inside ./config.py file. You may edit these files depending upon the location of datasets, ratio of how to split train/valid set, .., and so on.

Below is the training / testing steps for the one who wants to train the model via command line.  
There is also a ipynb file, so you can simply run the ipynb file easily. (But i highly recommanded to train via command line)

## Training

The param <model_name> is the folder name that will be created to store the whole model information.

```python
    CUDA_VISIBLE_DEVICES=0 python main.py --logdir <model_name>
```

Noted that default location of datasets in the config.py are `./data/dataset/train`.
Please make change accordingly.

## Testing

Like the param in training, the param <model_name> is the folder name that you want to use to evaluate the dataset.

```python
    CUDA_VISIBLE_DEVICES=0 python evaluate.py --model <model_name>
```

## Ensemble

Ensemble the prediction results of all models.

```python
    python ensemble_easy.py
```

# References
