# training-only-BN

This repository contains the final project of the DLAI (Deep Learning & Applied AI) course, a.y. 2021/2022, at Sapienza University of Rome.

### Specification 

Project 08: Training BatchNorm and only BatchNorm A CNN with weights frozen at their random original value and trained only on the beta and gamma parameters of the batch-normalization can achieve surprisingly good results on image classification problems, much better than training an equivalent number of weights chosen at random. With this project, you will investigate further this idea and: 1) discuss why this happens, 2) test experimentally this result on other architectures and other kinds of data, like MLPs or GNN, and 3) suggest further generalizations or applications that are not present in the current literature (to the best of your investigation). Reference: https://arxiv.org/abs/2003.00152

### Get started

The dependencies are stored in environment.txt. Create a virtual environment and install with 

```bash
pip install -r requirement.txt
```

### Code 
1) main.ipynb contains the main analysis of the project
2) utils_fuctions.py file that contains all implemented functions, e.g. training functions and plots. 

For convenience the notebook which can be viewed [here]()

### Dataset 

The dataset used in this project is the UPFD Dataset https://arxiv.org/abs/2104.12259
