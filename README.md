# PerFedRec++: Enhancing Personalized Federated Recommendation with Self-Supervised Pre-Training






![Static Badge](https://img.shields.io/badge/Paper-PDF-blue?style=flat&link=https%3A%2F%2Farxiv.org%2Fpdf%2F2305.06622.pdf)


This is the PyTorch implementation for PerFedRec++ model, an improved version of [PerFedRec](https://github.com/sichunluo/PerFedRec).


> **PerFedRec++: Enhancing Personalized Federated Recommendation with Self-Supervised Pre-Training.**  
Sichun Luo, Yuanzhang Xiao, Xinyi Zhang, Yang Liu, Wenbo Ding, Linqi Song.  
*ACM Transactions on Intelligent Systems and Technology (TIST) 2024*


---

## Introduction
In this paper, we propose PerFedRec++, to enhance the personalized federated recommendation with self-supervised pre-training. Specifically, we utilize the privacy-preserving mechanism of federated recommender systems to generate two augmented graph views, which are used as contrastive tasks in self-supervised graph learning to pre-train the model. Pre-training enhances the performance of federated models by improving the uniformity of representation learning. Also, by providing a better initial state for federated training, pre-training makes the overall training converge faster, thus alleviating the heavy communication burden. We then construct a collaborative graph to learn the client representation through a federated graph neural network. Based on these learned representations, we cluster users into different user groups and learn personalized models for each cluster. Each user learns a personalized model by combining the global federated model, the cluster-level federated model, and its own fine-tuned local model.

![Pre-Training](/fig/fig1.png)
![Training](/fig/fig2.png)

## Citation
If you find PerFedRec++ useful in your research or applications, please kindly cite:

> @article{luo2023perfedrec++,  
  title={PerFedRec++: Enhancing Personalized Federated Recommendation with Self-Supervised Pre-Training},  
  author={Luo, Sichun and Xiao, Yuanzhang and Zhang, Xinyi and Liu, Yang and Ding, Wenbo and Song, Linqi},  
  journal={arXiv preprint arXiv:2305.06622},  
  year={2023}  
}

Thanks for your interest in our work!


## Acknowledgement
Thanks for [SELFRec](https://github.com/Coder-Yu/SELFRec).
