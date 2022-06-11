from pytorch_lightning import Trainer
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import os
import pickle
from math import floor
from datetime import datetime
from libs.seq2seq_model import LinearRNNModel, LinearRNNEncDecModel, TransformerModel, LinearTransformerModel
from libs.lfgenerator import LinearRNNGroundTruth, TwoPart


def train_model(name, model, input, output, train_test_split, epochs=300, batch_size=128, check_point_monitor='valid_loss', devices=4):
    """_summary_

    Args:
        name (str): Name of this run
        model (Model):The model
        input (ndarray): input array
        output (ndarray): output array
        train_test_split (float): ratio of train test split
    """
    if input is not None:
    # If input not provided then skip this part
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input, dtype=torch.float32)
        if not isinstance(output, torch.Tensor):
            output = torch.tensor(output, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(input, output)
        total = len(dataset)
        train_size = floor(total*train_test_split)
        test_size = total - train_size

        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,drop_last=True, num_workers=os.cpu_count(), pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,drop_last=True, num_workers=os.cpu_count(), pin_memory=True)
    else:
        train_loader = None
        valid_loader = None

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    now = datetime.now().strftime("%H:%M:%S__%m-%d")
    checkpoint_callback = ModelCheckpoint(
                                        save_top_k=4, 
                                        monitor=check_point_monitor,
                                        filename=name + "-{epoch:02d}-{valid_loss:.2e}") 
    
    if devices == 1:
        trainer = Trainer(accelerator="gpu", 
                    log_every_n_steps=30,
                    devices=1,
                    max_epochs=epochs,
                    precision=32,
                    logger=TensorBoardLogger("runs", name=name),
                    callbacks=[checkpoint_callback, lr_monitor])
    else:
        trainer = Trainer(accelerator="gpu", 
                    log_every_n_steps=30,
                    devices=devices,
                    strategy=DDPStrategy(find_unused_parameters=False),
                    max_epochs=epochs,
                    precision=32,
                    logger=TensorBoardLogger("runs", name=name),
                    callbacks=[checkpoint_callback, lr_monitor])

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)



def train_rnn():
    hid_dim = 128
    model = LinearRNNModel(hid_dim=hid_dim, input_dim=1, output_dim=1)
    U = model.rnn.input_ff._parameters['weight']
    W = model.rnn.hidden_ff._parameters['weight']
    cT = model.rnn.output_ff._parameters['weight']
    cT.data = torch.ones_like(cT.data)*1.0
    W.data = torch.diag(torch.rand(hid_dim))
    # W.data = torch.diag(torch.tensor([0.933032,0.95020021]))
    U.data = torch.ones_like(U.data)*1.0

    input, output =  TwoPart({'input_dim': 1,
                                            'output_dim': 1,
                                            'data_num': 1280000,
                                            'path_len': 32,}).generate()

    train_model('rnn', model, input, output, 0.8, batch_size=1024, epochs=5000, devices=4)


def train_rnnencdec():
    hid_dim = 32
    seq_len = 64
    model = LinearRNNEncDecModel(hid_dim=hid_dim, input_dim=1, output_dim=1, out_len=seq_len)
    U = model.rnn.U._parameters['weight']
    W = model.rnn.W._parameters['weight']
    cT = model.rnn.cT._parameters['weight']
    V = model.rnn.V._parameters['weight']
    M = model.rnn.M._parameters['weight']

    cT.data = torch.ones_like(cT.data)*1.0
    W.data = torch.diag(torch.rand(hid_dim))
    M.data = torch.diag(torch.rand(hid_dim))
    V.data = torch.diag(torch.rand(hid_dim))
    U.data = torch.ones_like(U.data)*1.0


    input, output =  TwoPart({'input_dim': 1,
                            'output_dim': 1,
                            'data_num': 1280000//2,
                            'path_len': seq_len ,
                            'centers': [6, 25, 50],
                            'sigmas':[0.5, 0.5, 0.2]}).generate()

    train_model('rnnEncDec', model, input, output, 0.8, batch_size=512, epochs=5000, devices=[0,1,2,3])



# def train_transformer():
#     hid_dim = 64
#     seq_len = 64
#     model =TransformerModel(input_dim=1, output_dim=1, num_layers=1, hid_dim=hid_dim, nhead=8, src_length=seq_len )



#     input, output =  TwoPart({'input_dim': 1,
#                             'output_dim': 1,
#                             'data_num': 1280000//2,
#                             'path_len': seq_len ,
#                             'centers': [6, 25, 50],
#                             'sigmas':[0.5, 0.5, 0.2]}).generate()

#     train_model('transformer', model, input, output, 0.8, batch_size=512, epochs=5000, devices=[0,2,3])


def train_LinearTrans_multiple_part(data, exp_name, attention_type, layer_norm, attn_norm, norm_type, devices):

    model = LinearTransformerModel(input_dim=1, output_dim=1, num_layers=5,hid_dim=256, nhead=16,src_length=64, 
                                    attention_type=attention_type, layer_norm=layer_norm,
                                     attn_norm=attn_norm, norm_type=norm_type)
    input, output = data

    train_model(exp_name, model, input, output, 0.8, batch_size=512,  epochs=2000, devices=devices)