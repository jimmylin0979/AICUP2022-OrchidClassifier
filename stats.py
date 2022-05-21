#
import fire

#
import torch
import torch.nn as nn
#
import numpy as np

#
from tqdm import tqdm

#
import models
from config import DefualtConfig

#
from fvcore.nn import FlopCountAnalysis

###################################################################################

config = DefualtConfig()
device = torch.device(f'cuda:{config.use_gpu_index}' if torch.cuda.is_available(
) else'cpu') if config.use_gpu_index != -1 else torch.device('cpu')


def main():

    model = getattr(models, config.model_name)(config)
    model.to(device)

    flops = FlopCountAnalysis(model, torch.zeros((1, 3, 224, 224)).to(device))
    print(f'FLOPs : {flops.total()}')
    print(f'Operators : \n{flops.by_operator()}')

    #     # Metrics : FLOPs, Params
    # with torch.cuda.device(0):
    #     macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=False, verbose=False)
    #     logger.log('pthflops : ')
    #     logger.log('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     logger.log('{:<30}  {:<8}'.format('Number of parameters: ', params))


if __name__ == '__main__':
    main()
