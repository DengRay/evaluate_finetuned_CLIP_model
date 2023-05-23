import torch
import torch.nn as nn
import numpy as np
model_path = '/home/dengyiru/Chinese-CLIP-master/data_path/experiments/vip_finetune_vit-b-16_roberta-base_bs128_4gpu/checkpoints/epoch_latest.pt'
#smodel_path = '/home/dengyiru/Chinese-CLIP-master/data_path/pretrained_weights/clip_cn_vit-b-16.pt'
model = torch.load(model_path)
model_keys = model.keys()
name = model['state_dict']
print(type(name))
#print(len(name[1]))
for key, value in name.items():
    print(key)
'''
def print_model_structure(model, indent=0):
    for name, module in model.named_children():
        print(" " * indent, name)
        if len(list(module.children())) > 0:
            print_model_structure(module, indent + 2)

print_model_structure(name)
'''


