# The code of DR-GNN
You can get the results of different datasets in the paper by running the following code. We use pytorch==2.0.1 version in the paper.
```
nohup python main.py --model=lgn --enable_DRO=1 --aug_on --full_batch --ood popularity_shift --dataset='gowalla' --weight_decay 0.0001 --alpha 0.04 --tau 1  --aug_coefficient 0.1 --aug_ratio 0.2 --cuda 6 &
```
```
nohup python main.py --model=lgn --enable_DRO=1 --aug_on --full_batch --ood popularity_shift --dataset='douban' --weight_decay 1e-7 --alpha 0.005 --tau 0.1 --aug_coefficient 0.04 --aug_ratio 0.05 --cuda 6 &
```
```
nohup python main.py --model=lgn --enable_DRO=1 --aug_on --full_batch --ood popularity_shift --dataset='amazon-book' --weight_decay 0.0001 --alpha 0.03 --tau 0.8 --aug_coefficient 0.1 --aug_ratio 0.1 --cuda 7 &
```
```
nohup python main.py --model=lgn --enable_DRO=1 --aug_on --full_batch --ood popularity_shift --dataset='yelp2018' --weight_decay 0.0001 --alpha 0.07 --tau 1 --aug_coefficient 0.25 --aug_ratio 0.05 --cuda 7 &
```
