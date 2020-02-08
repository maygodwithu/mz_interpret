#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import pandas as pd
import matchzoo as mz
print('matchzoo version', mz.__version__)


# In[2]:


ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=1))
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]
print("`ranking_task` initialized with metrics", ranking_task.metrics)


# In[3]:


print('data loading ...')
train_pack_raw = mz.datasets.wiki_qa.load_data('train', task=ranking_task)
dev_pack_raw = mz.datasets.wiki_qa.load_data('dev', task=ranking_task, filtered=True)
test_pack_raw = mz.datasets.wiki_qa.load_data('test', task=ranking_task, filtered=True)
print('data loaded as `train_pack_raw` `dev_pack_raw` `test_pack_raw`')


# In[4]:


preprocessor = mz.models.MatchPyramid.get_default_preprocessor()


# In[5]:


train_pack_processed = preprocessor.fit_transform(train_pack_raw)
dev_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)


# In[6]:


preprocessor.context


# In[7]:


glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=300)
term_index = preprocessor.context['vocab_unit'].state['term_index']
embedding_matrix = glove_embedding.build_matrix(term_index)
l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]


# In[8]:


trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='pair',
    num_dup=2,
    num_neg=1,
    batch_size=20,
    resample=True,
    sort=False,
    shuffle=True
)
testset = mz.dataloader.Dataset(
    data_pack=test_pack_processed,
    batch_size=20,
    sort=False,
    shuffle=False
)


# In[9]:


padding_callback = mz.models.MatchPyramid.get_default_padding_callback()

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


# In[10]:


model = mz.models.MatchPyramid()

model.params['task'] = ranking_task
model.params['embedding'] = embedding_matrix
model.params['kernel_count'] = [16, 32]
model.params['kernel_size'] = [[3, 3], [3, 3]]
model.params['dpool_size'] = [3, 10]
model.params['dropout_rate'] = 0.1

model.build()

print(model)
print('Trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))


# In[11]:


optimizer = torch.optim.Adam(model.parameters())

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=testloader,
    validate_interval=None,
    save_dir='save_match_pyramid',
    epochs=5
)


# In[12]:


trainer.run()

