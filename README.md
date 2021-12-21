# deep_nextitnet

## **1. Purpose**

Model compression for NextItNet

NextItNet: https://github.com/fajieyuan/WSDM2019-nextitnet


Idea from Bert of Theseus

https://github.com/JetRunner/BERT-of-Theseus

## **2. Coding file explanation**

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


## **3. How to do compression**

1. Train a predecessor model:
```bash
#
python ./deeprec_pretrained.py \  
  --top_k 5 \
  --modelpath /path/to/save_successor/ \
  --dataset  /name of dataset \
  --datapath /path/to/dataset \
  --eval_iter 3000 \
  --save_para_every 3000 \
  --tt_percentage  0.1\
```

2. Run compression following the examples below:
```bash
# For compression with a replacement scheduler
python ./deeprec_replacing.py \
  --top_k 5\
  --premodel /path/to/saved_predecessor \
  --dataset  /name of dataset \
  --datapath /path/to/dataset \
  --eval_iter 3000 \
  --save_para_every 3000 \
  --modelpath /path/to/save_successor/ \
  --tt_percentage  0.1\
  --scheduler_linear_k 0.0006
```
For the detailed description of arguments, please refer to the source code.

