# deep_nextitnet

**1. Purpose**

Model compression for NextItNet

NextItNet: https://github.com/fajieyuan/WSDM2019-nextitnet


Idea from Bert of Theseus

https://github.com/JetRunner/BERT-of-Theseus

**2. Coding file explanation**

deeprec_pretrain.py          
Train a predecessor Model

deeprec_replacing.py          
Train a small successor model(using 'linear' or 'constant' replacing rate)

generator_deep.py  
Build a computing graph for the predecessor model

generator_deep_replacing.py   
Build a computing graph for replacing stage

suc_graph.py  
Build a computing graph for successor model

data_pretrain.py  
Loading a dataset

ops.py  
Build residual blocks and compute forward propogation result
