import pytorch_lightning as pl
import torch
from torch import nn
from libs.layers import LinearRNN, LinearRNNEncDec, ConstantPositionalEncoding

class Seq2SeqModel(pl.LightningModule):
    def __init__(self):
        super().__init__() 
        
        self.save_hyperparameters()

    def forward(self, x):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = {
                        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer), 
                        "interval": "epoch",
                        "frequency": 1,
                        "monitor": "train_loss_epoch"
                    }
        return {"optimizer": optimizer, "lr_scheduler":scheduler}

    def training_step(self, batch, batch_idx, loss=nn.MSELoss()):
        x, y = batch
        y_hat = self(x)
        trainloss = loss(y_hat, y)
        self.log("train_loss", trainloss, on_epoch=True, prog_bar=True, logger=True)
        return trainloss

    def validation_step(self, batch, batch_idx, loss=nn.MSELoss()):
        x, y = batch
        y_hat = self(x)
        validloss = loss(y_hat, y)
        self.log("valid_loss", validloss, prog_bar=True, logger=True)
        return validloss

    def predict(self, x, return_extra=False):
        self.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)

            if return_extra:
                pred, extra = self(x, True)
                return pred.detach().cpu().numpy(), extra
            else:
                pred = self(x)
                return pred.detach().cpu().numpy()
                


class LinearRNNModel(Seq2SeqModel):
    def __init__(self, hid_dim, input_dim, output_dim):
        super().__init__()
        self.rnn = LinearRNN(input_dim=input_dim, output_dim=output_dim, hid_dim=hid_dim)

    def forward(self, x):
        y = self.rnn(x)

        return y

class LinearRNNEncDecModel(Seq2SeqModel):
    def __init__(self, hid_dim, input_dim, output_dim, out_len):
        super().__init__()
        self.rnn = LinearRNNEncDec(input_dim=input_dim, output_dim=output_dim, hid_dim=hid_dim, out_len=out_len)

    def forward(self, x):
        y = self.rnn(x)

        return y


class TransformerModel(Seq2SeqModel):
    def __init__(self, input_dim, output_dim, num_layers, hid_dim, nhead, src_length, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.input_ff = nn.Linear(input_dim, hid_dim)
        self.output_ff =  nn.Linear(hid_dim, output_dim)
        transformerlayer = nn.TransformerEncoderLayer(d_model=hid_dim,dim_feedforward=hid_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformerlayer, num_layers=num_layers)
        self.pos_encoder = ConstantPositionalEncoding(hid_dim, max_len=src_length)
        mask = self._generate_square_subsequent_mask(src_length)
        self.register_buffer('mask', mask)
        pos_embedding= nn.Parameter(torch.zeros(1, src_length, hid_dim))
        self.register_buffer('pos_embedding', pos_embedding)
    def forward(self, x):
        x = self.input_ff(x)
        x = self.pos_encoder(x)
        # x = x + self.pos_embedding
        y = self.transformer(x)
        output = self.output_ff(y)
        
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class LinearTransformerModel(Seq2SeqModel):
    def __init__(self, input_dim, output_dim, num_layers, hid_dim, nhead, src_length, 
                    attention_type, layer_norm, attn_norm, norm_type, dropout=0.05):
        super(LinearTransformerModel, self).__init__()
        import copy
        from libs.galerkin_transformer.libs.model import SimpleTransformerEncoderLayer

        encoder_layer = SimpleTransformerEncoderLayer(
                        d_model=hid_dim,
                        pos_dim=0,
                        n_head=nhead,
                        dim_feedforward=hid_dim,
                        attention_type=attention_type,
                        layer_norm=layer_norm,
                        attn_norm=attn_norm,
                        norm_type=norm_type,
                        attn_weight=True,
                        dropout=dropout)
        self.encoder_layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.input_ff = nn.Linear(input_dim, hid_dim)
        self.output_ff =  nn.Linear(hid_dim, output_dim)
       
       
        self.pos_encoder = ConstantPositionalEncoding(hid_dim, max_len=src_length)

    def forward(self, x, return_attn=False):
        x = self.input_ff(x)
        x = self.pos_encoder(x)
        for layer in self.encoder_layers:
            x, attn = layer(x)
        output = self.output_ff(x)
        
        if return_attn:
            return output, attn
        else:
            return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask