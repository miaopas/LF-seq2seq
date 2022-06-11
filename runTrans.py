from libs.train import *
from libs.utils import product_dict

d = {'attention_type':['fourier', 'softmax'], 
    'layer_norm':[True, False], 
    'attn_norm':[False,True],
    'norm_type': ['layer']}
options = [{'exp_name':f'multiple_part-galerkin{i}'}|options for i, options in enumerate(product_dict(**d), start=0)]
for option in options:
    if not option['attn_norm']:
        option['norm_type'] = None

print(options[4])


data = TwoPart({'input_dim': 1,
                            'output_dim': 1,
                            'data_num': 1280000//2,
                            'path_len': 64 ,
                            'centers': [6, 25, 50],
                            'sigmas':[0.5, 0.5, 0.2]}).generate()


train_LinearTrans_multiple_part(data, devices=4, **options[4])
# for i in range(4):
#     print(f'Current in run {i}')
#     for option in options:
#         print(f'Current on {option}\n')

#         train_LinearTrans_multiple_part(data, devices=4, **option)