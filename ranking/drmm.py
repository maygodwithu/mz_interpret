#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('run', 'init.ipynb')
import torch
import numpy as np
import pandas as pd
import matchzoo as mz
print('matchzoo version', mz.__version__)


# In[2]:
ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=10))
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]

# In[3]:

print('data loading ...')
train_pack_raw = mz.datasets.wiki_qa.load_data('train', task=ranking_task)
dev_pack_raw = mz.datasets.wiki_qa.load_data('dev', task=ranking_task, filtered=True)
test_pack_raw = mz.datasets.wiki_qa.load_data('test', task=ranking_task, filtered=True)
print('data loaded as `train_pack_raw` `dev_pack_raw` `test_pack_raw`')


preprocessor = mz.models.DRMM.get_default_preprocessor()


# In[4]:


train_pack_processed = preprocessor.fit_transform(train_pack_raw)
dev_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)


# In[5]:


preprocessor.context


# In[6]:


glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=300)
term_index = preprocessor.context['vocab_unit'].state['term_index']
embedding_matrix = glove_embedding.build_matrix(term_index)
l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]


# In[7]:


histgram_callback = mz.dataloader.callbacks.Histogram(
    embedding_matrix, bin_size=30, hist_mode='LCH'
)

trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='pair',
    batch_size=20,
    resample=True,
    num_dup=5,
    num_neg=10,
    callbacks=[histgram_callback]
)
testset = mz.dataloader.Dataset(
    data_pack=test_pack_processed,
    batch_size=20,
    callbacks=[histgram_callback]
)


# In[8]:


padding_callback = mz.models.DRMM.get_default_padding_callback()

trainloader = mz.dataloader.DataLoader(
    device='cpu',
    dataset=trainset,
    stage='train',
    callback=padding_callback
)
testloader = mz.dataloader.DataLoader(
    dataset=testset,
    stage='dev',
    callback=padding_callback
)


# In[9]:


model = mz.models.DRMM()

model.params['task'] = ranking_task
model.params['mask_value'] = 0
model.params['embedding'] = embedding_matrix
model.params['hist_bin_size'] = 30
model.params['mlp_num_layers'] = 1
model.params['mlp_num_units'] = 10
model.params['mlp_num_fan_out'] = 1
model.params['mlp_activation_func'] = 'tanh'

model.build()

print(model)
print('Trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))


# In[10]:


optimizer = torch.optim.Adadelta(model.parameters())

trainer = mz.trainers.Trainer(
    device='cpu',
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=testloader,
    validate_interval=None,
    save_dir='save_drmm',
    epochs=10
)


# In[11]:


trainer.run()

