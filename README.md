# Fully-linear DenseNet
Implementation of a Fully-linear DenseNet for pipe busrt location of a water distribution network.

## Some cmd arguments:
*    --arch  choose a kind of architecture to use
*    --out   the file name of predict result
*    --lr    learning rate 
*    --epoch training epochs
*    --kind   the name of the log file recording information during running
*    --lr_decay learning rate decay
*    --data_dir the path of dataset
*    --save_path the path of saving model (need to change code)
*    --num_init_features an argument of arch waterdsnetf_self_define 
*    --growth_rate an argument of arch waterdsnetf_self_define


### example cmd 1 (default Anytown dataset) :
python train.py main --arch waterdsnetf --lr 0.6 --epoch 120 --kind Anytown_P10C10_B0105_4M --data_dir ~/water/Anytown_P10C10_B0105_4M  --save_path './water/modelparams' --out Anytown_P10C10_B0105_4M

### example cmd 2 (dataset Anytown with Duration changed) :
python train.py main --arch waterdsnetf_self_define --num_init_features 320 --growth_rate 32 --lr 0.6 --epoch 120 --kind Anytown_P10C10_B1030_Duration5 --data_dir ~/water/Anytown_P10C10_B1030_Duration5   --save_path './water/modelparams' --out Anytown_P10C10_B1030_Duration5 

### example cmd 3 (Mudu dataset) ：
CUDA_VISIBLE_DEVICES=0 python train.py main --arch waterdsnetf_in4_out58 --lr 0.6 --epoch 120 --kind Mudu_P10C10_B1030_4M --data_dir ~/water/Mudu_P10C10_B1030_4M  --save_path './water/modelparams' --out Mudu_P10C10_B1030_4M 

## Tricks during training

* diminish learning rate with traning epochs increasing
```
You can directly changde the codes to set the periods of diminishing.
```
* choose a best model with lowest loss since the last time when learning rate changed 
```
When the learning rate changes, we will choose a set of parameters of the best model, of which the loss is the minimum since the last time learning rate changes. After that, the current parameters is replaced with the chosen ones. 
In order to train faster and save space, the procedure can be omitted. As the corresponding parts is commented-out, the readers can read and reuse it if they think this trick useful.
```

## other super-parameters:
* batch_size
```
Default batch_size is 128, you can change it in the train.py
```
* weight_decay
```
Default weight_decay is 0.00005, you can change it in config.py
```


