import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
from .mynn.my_emodule import my_ReLU, my_MaxPool2d, my_Conv2d, my_Linear, my_BatchNorm2d, my_AvgPool2d, my_Mean

import typing
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel
from matchzoo.dataloader import callbacks
from matchzoo.modules import Attention

#class SNRM(nn.Module):
class SNRM(BaseModel):
    """
    Stand alone neural ranking
    """
    def __deprecated_init__(self, args):
        super(SNRM, self).__init__()
        self.update_lr = args.learning_rate
        self.dropout_r = args.dropout_parameter
        self.regularization = args.regularization_term
        self.emb_dim = args.emb_dim
        self.conv1_ch = args.conv1_channel
        self.conv2_ch = args.conv2_channel
        self.conv3_ch = args.conv3_channel

        ## make network
        self.features = self._make_layers()

        ## hinge loss
        self.loss = nn.HingeEmbeddingLoss()
#        self.loss = nn.MarginRankingLoss()

        ## optimizer
        self.optimizer = optim.Adam(self.parameters(), lr = args.learning_rate)

        ## mandatory for path
        self._layers = None 

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(
            with_embedding=True,
            with_multi_layer_perceptron=False
        )
        params.add(Param(name='learning_rate', value=0.00001,
                         desc="learning_rate"))
        params.add(Param(name='regularization', value=0.0000001,
                         desc="regularization"))
        params.add(Param(name='conv1_channel', value=500,
                         desc="conv1_channel"))
        params.add(Param(name='conv2_channel', value=300,
                         desc="conv2_channel"))
        params.add(Param(name='conv3_channel', value=5000,
                         desc="conv3_channel"))
        params.add(Param(name='emb_dim', value=300,
                         desc="emb_dim"))
        return params

    @classmethod
    def get_default_padding_callback(
        cls,
        fixed_length_left: int = None,
        fixed_length_right: int = None
    ):
        """:return: Default padding callback."""
        return callbacks.BasicPadding(
            fixed_length_left=fixed_length_left,
            fixed_length_right=fixed_length_right,
        )

    def build(self):
        """Build model structure."""
        self.embedding = self._make_default_embedding_layer()
        #print('embedding=', self.embedding)
        #print('emb test=', self.embedding(torch.tensor([1]).long()))
        self.mlp = self._make_layers(
            self._params['conv1_channel'],
            self._params['conv2_channel'],
            self._params['conv3_channel'],
            self._params['emb_dim']
        )
        self.out = my_Mean(dim=(2,3))

    def _make_layers(self, conv1_ch, conv2_ch, conv3_ch, emb_dim): 
        layers = []
        layers += [my_Conv2d(emb_dim, conv1_ch, kernel_size=(1,5),padding=(0,2))]
        layers += [my_ReLU(inplace=True)]
        #layers += [nn.Dropout(p=self.dropout_r, inplace=True)]
        layers += [my_Conv2d(conv1_ch, conv2_ch, kernel_size=1)]
        layers += [my_ReLU(inplace=True)]
        #layers += [nn.Dropout(p=self.dropout_r, inplace=True)]
        layers += [my_Conv2d(conv2_ch, conv3_ch, kernel_size=1)]
        layers += [my_ReLU(inplace=True)]
        #layers += [nn.Dropout(p=self.dropout_r, inplace=True)]
        #layers += [my_Mean(dim=(2,3))]
        return nn.Sequential(*layers)

    #def model_train(self, query, doc1, doc2, label):
    #def forward_with_regulizer(self, inputs):
#        query, doc = inputs['text_left'], inputs['text_right']
# 
#        embed_query = self.embedding(query.long())
#        embed_doc = self.embedding(doc.long())
#
#        eqs = embed_query.shape
#        eds = embed_doc.shape
#        embed_query = embed_query.transpose(1,2).reshape(eqs[0],eqs[2],1,eqs[1])
#        embed_doc = embed_doc.transpose(1,2).reshape(eds[0],eds[2],1,eds[1])
    def _make_input(self, inputs):
        query, doc = inputs['text_left'], inputs['text_right']
 
        embed_query = self.embedding(query.long())
        embed_doc = self.embedding(doc.long())

        embed_query = embed_query.transpose(1,2).unsqueeze(2)
        embed_doc = embed_doc.transpose(1,2).unsqueeze(2)

        return embed_query, embed_doc


    def forward_with_regulizer(self, inputs):
        embed_query, embed_doc = self._make_input(inputs)
        q_repr = self._one_forward(embed_query.float())
        d1_repr = self._one_forward(embed_doc.float())
        logits_d1 = torch.sum(q_repr * d1_repr, 1, keepdim=True) 

        L1_regul = self._params['regularization'] * (torch.norm(q_repr, p=1) + torch.norm(d1_repr, p=1))
  
        return logits_d1, L1_regul

    def forward(self, inputs):
        logits_d1, L1_regul = self.forward_with_regulizer(inputs)
        return logits_d1

    def grad_forward(self, inputs):
        embed_query, embed_doc = self._make_input(inputs) 
        q = Variable(embed_query, requires_grad=True)
        d = Variable(embed_doc, requires_grad=True)

        q_repr = self._one_forward(q.float())
        d1_repr = self._one_forward(d.float())
        out = torch.sum(q_repr * d1_repr, 1, keepdim=True) 

        out[0].backward()

        #print(out.shape)
        #print(q.shape)
        #print(d.shape)

        q_multi = q * q.grad
        d_multi = d * d.grad

        outputs = {'score' : out[0],
                   'qgrad' : self._ct(q.grad),
                   'dgrad' : self._ct(d.grad),
                   'qmulti' : self._ct(q_multi),
                   'dmulti' : self._ct(d_multi),
                   'qrepr' : q_repr,
                   'drepr' : d1_repr}

        return outputs

    def gradcam_forward(self, inputs):
        embed_query, embed_doc = self._make_input(inputs) 

        q_act = self.mlp(embed_query.float())
        d_act = self.mlp(embed_doc.float())

        q = Variable(q_act, requires_grad=True)
        d = Variable(d_act, requires_grad=True)

        q_repr = self.out(q)
        d1_repr = self.out(d)

        out = torch.sum(q_repr * d1_repr, 1, keepdim=True) 

        out[0].backward()

        q_grad_val = q.grad.cpu().data.numpy()
        d_grad_val = d.grad.cpu().data.numpy()

        q_act_val = q_act.cpu().data.numpy()[0, :] 
        d_act_val = d_act.cpu().data.numpy()[0, :] 

        qweights = np.mean(q_grad_val, axis=(2, 3))[0, :]
        qcam = np.zeros(q_act_val.shape[1:], dtype=np.float32)
        for i, w in enumerate(qweights):
            qcam += w * q_act_val[i, :, :] 

        dweights = np.mean(d_grad_val, axis=(2, 3))[0, :]
        dcam = np.zeros(d_act_val.shape[1:], dtype=np.float32)
        for i, w in enumerate(dweights):
            dcam += w * d_act_val[i, :, :] 

        outputs = {'score' : out[0],
                   'qgrad' : q_grad_val,
                   'dgrad' : d_grad_val,
                   'qcam' : torch.from_numpy(qcam),
                   'dcam' : torch.from_numpy(dcam),
                   'qrepr' : q_repr,
                   'drepr' : d1_repr}

        return outputs

      
    def _one_forward(self, x):
        out = self.mlp(x)
        return self.out(out)

    ## change into original type
    def _ct(self, x):
        return x.squeeze(2).transpose(1,2)

    ###!! Mandatory functions
    # fill_layers()
    def fill_layers(self, x):
        self._layers = []

        fe = self.out
        name = fe._get_name()
        shape = fe.getOutShape()
        self._layers.append((name, shape, fe))

        for fe in reversed(self.mlp): ## backward by the reversed order
            name = fe._get_name()
            if('Dropout' in name): continue
            shape = fe.getOutShape()
            self._layers.append((name, shape, fe))
            print(name, shape)

        self._layers.append(('Input', x.shape, None))
 
