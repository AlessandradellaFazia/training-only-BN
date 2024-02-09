# training-only-BN

This repository contains the final project of the [DLAI](https://github.com/erodola/DLAI-s2-2022) (Deep Learning & Applied AI) course, a.y. 2021/2022, at Sapienza University of Rome.

### Specification 

Project 08: Training BatchNorm and only BatchNorm A CNN with weights frozen at their random original value and trained only on the beta and gamma parameters of the batch-normalization can achieve surprisingly good results on image classification problems, much better than training an equivalent number of weights chosen at random. With this project, you will investigate further this idea and: 1) discuss why this happens, 2) test experimentally this result on other architectures and other kinds of data, like MLPs or GNN, and 3) suggest further generalizations or applications that are not present in the current literature (to the best of your investigation). Reference: https://arxiv.org/abs/2003.00152

### Get started

The dependencies are stored in environment.txt. Create a virtual environment and install with 

```bash
pip install -r requirement.txt
```

### Content 
1) **main.ipynb** contains the main analysis of the project
2) **utils_fuctions.py** file that contains all implemented functions, e.g. training functions and plots.
3) environment.txt for recreate the virtual python environment. 

For convenience the notebook can be viewed [here](https://nbviewer.org/github/AlessandradellaFazia/training-only-BN/blob/main/main.ipynb)

### Dataset 

The dataset used in this project is the UPFD Dataset https://arxiv.org/abs/2104.12259


### Purpose 

In this project we test experimentally how it is possible to obtain good results by training only the parameters of batch normalization on non-Euclidean data such as graphs.
The chosen problem was Graph Classification for discern fake news from real news.
Traditional methods for detecting fake news is fact-checking that required time-consuming work for acquire evidence from domain experts.
In the above-mentioned work instead, user preferences (personality, sentimentent and stance) are used together with the propagation graph of the news, in addition to the textual content of the news. The user's preference are estimated by looking at the historical posts. News propagation graph is build on the chain of retweets of a news.
[pythorch geometric](https://pytorch-geometric.readthedocs.io/en/stable/index.html) was used to handle the graphs neural networks. 
