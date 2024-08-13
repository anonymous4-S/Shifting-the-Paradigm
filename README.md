# Shifting the Paradigm: A Diffeomorphism Between Time Series Data Manifolds for Achieving Shift-Invariancy in Deep Learning

Running
----------
All the requested baselines are integrated into the run command where only the backbone should be changed.

```
python main_supervised_baseline.py --dataset 'ucihar' --backbone 'ModernTCN' --lr 5e-5 --n_epoch 999 --cuda 0
python main_supervised_baseline.py --dataset 'ucihar' --backbone 'WaveletNet' --lr 5e-5 --n_epoch 999 --cuda 0
```


To investigate the robustness of the models,

```
python main_supervised_baseline.py --dataset 'ucihar' --backbone 'ModernTCN' --lr 5e-5 --n_epoch 999 --cuda 0 --robust_check
python main_supervised_baseline.py --dataset 'ucihar' --backbone 'WaveletNet' --lr 5e-5 --n_epoch 999 --cuda 0 --robust_check
```


To observe the effect of the proposed transformation with the guidance network,

```
python main_supervised_baseline.py --dataset 'ucihar' --backbone 'ModernTCN' --lr 5e-5 --n_epoch 999 --cuda 0 --controller
python main_supervised_baseline.py --dataset 'ucihar' --backbone 'WaveletNet' --lr 5e-5 --n_epoch 999 --cuda 0 --controller
```


To observe the effect of the proposed transformation to the shift-invariancy, *--robust_check* command can be added,

```
python main_supervised_baseline.py --dataset 'ucihar' --backbone 'ModernTCN' --lr 5e-5 --n_epoch 999 --cuda 0 --controller --robust_check
```



 
