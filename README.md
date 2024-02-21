# Distributionally Robust Graph-based Recommendation System (DR-GNN)
This is the PyTorch implementation for our WWW 2024 paper. 
> Bohao Wang, Jiawei Chen, Changdong Li, Sheng Zhou, Qihao Shi, Yang Gao, Yan Feng, Chun Chen, Can Wang 2024. Distributionally Robust Graph-based Recommendation System. [arXiv link](https://arxiv.org/abs/2402.12994)

## Requirements
To install requirements:
```
pip install -r requirements.txt
```

## Training & Evaluation
You can get the results in the paper by running the following code.

**Gowalla**
```
python main.py --model=lgn --enable_DRO=1 --aug_on --full_batch --ood popularity_shift --dataset='gowalla' --weight_decay 0.0001 --alpha 0.04 --tau 1  --aug_coefficient 0.1 --aug_ratio 0.2
```
**Douban**
```
python main.py --model=lgn --enable_DRO=1 --aug_on --full_batch --ood popularity_shift --dataset='douban' --weight_decay 1e-7 --alpha 0.005 --tau 0.1 --aug_coefficient 0.04 --aug_ratio 0.05
```
**Amazon Book**
```
python main.py --model=lgn --enable_DRO=1 --aug_on --full_batch --ood popularity_shift --dataset='amazon-book' --weight_decay 0.0001 --alpha 0.03 --tau 0.8 --aug_coefficient 0.1 --aug_ratio 0.1
```
**Yelp2018**
```
python main.py --model=lgn --enable_DRO=1 --aug_on --full_batch --ood popularity_shift --dataset='yelp2018' --weight_decay 0.0001 --alpha 0.07 --tau 1 --aug_coefficient 0.25 --aug_ratio 0.05
```

## Citation
If you find the paper useful in your research, please consider citing:
```
@misc{wang2024distributionally,
      title={Distributionally Robust Graph-based Recommendation System}, 
      author={Bohao Wang and Jiawei Chen and Changdong Li and Sheng Zhou and Qihao Shi and Yang Gao and Yan Feng and Chun Chen and Can Wang},
      year={2024},
      eprint={2402.12994},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```

## Acknowledgments

This project makes use of the following open source projects:

- [LightGCN-PyTorch](https://github.com/gusye1234/LightGCN-PyTorch/tree/master): Some of the functionalities are inspired by this project. Thanks for their great work.
