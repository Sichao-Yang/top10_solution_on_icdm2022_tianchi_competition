# Introduction

this repo contains my solution to [ICDM 2022 : Risk Commodities Detection on Large-Scale E-commerce Graphs](https://tianchi.aliyun.com/competition/entrance/531976/introduction?lang=en-us)

it ranked 5th for session 1 and 10th for session 2.

Key methods used in this solution are:

* NARS
* SCR
* Meta Pseudo Label
* GAMLP
* focal loss

# Setup

run `pip install -r requirement.txt`, dgl library is used.

# Run experiment

solution for seesion 1: `python main.py -c config/dgl_session1.yml`

solution for session 2: `python main.py -c config/dgl_session2.yml`

dataset for this competition:

链接：https://pan.baidu.com/s/1gE-kNJ1fr-cFd1blnchtQw 
提取码：rzdy