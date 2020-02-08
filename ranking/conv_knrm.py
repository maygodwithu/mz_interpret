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
ranking_task = mz.tasks.Ranking(losses=mz.losses.RankHingeLoss())
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]


print('data loading ...')
train_pack_raw = mz.datasets.wiki_qa.load_data('train', task=ranking_task)
dev_pack_raw = mz.datasets.wiki_qa.load_data('dev', task=ranking_task, filtered=True)
test_pack_raw = mz.datasets.wiki_qa.load_data('test', task=ranking_task, filtered=True)
print('data loaded as `train_pack_raw` `dev_pack_raw` `test_pack_raw`')



preprocessor = mz.models.ConvKNRM.get_default_preprocessor()

# In[3]:


train_pack_processed = preprocessor.fit_transform(train_pack_raw)
dev_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)


# In[4]:


preprocessor.context


# In[5]:


glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=300)
term_index = preprocessor.context['vocab_unit'].state['term_index']
embedding_matrix = glove_embedding.build_matrix(term_index)
l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]


# In[6]:


trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='pair',
    batch_size=20,
    resample=True,
    sort=False,
    num_dup=5,
    num_neg=1
)
testset = mz.dataloader.Dataset(
    data_pack=test_pack_processed,
    batch_size=20,
)


# In[7]:


padding_callback = mz.models.ConvKNRM.get_default_padding_callback()

trainloader = mz.dataloader.DataLoader(
    dataset=trainset,
    stage='train',
    callback=padding_callback
)
testloader = mz.dataloader.DataLoader(
    dataset=testset,
    stage='dev',
    callback=padding_callback
)


# In[8]:


model = mz.models.ConvKNRM()

model.params['task'] = ranking_task
model.params['embedding'] = embedding_matrix
model.params['filters'] = 128 
model.params['conv_activation_func'] = 'tanh' 
model.params['max_ngram'] = 3
model.params['use_crossmatch'] = True 
model.params['kernel_num'] = 11
model.params['sigma'] = 0.1
model.params['exact_sigma'] = 0.001

model.build()

print(model)
print('Trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))


# In[9]:


optimizer = torch.optim.Adadelta(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3)

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=testloader,
    validate_interval=None,
    save_dir='save_conv_knrm',
    epochs=10,
    scheduler=scheduler,
    clip_norm=10
)


# In[10]:


trainer.run()

