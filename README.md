# Awesome Resources on Trustworthy Graph Neural Networks

![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![Stars](https://img.shields.io/github/stars/Radical3-HeZhang/Awesome-Trustworthy-GNNs?color=yellow)  ![Forks](https://img.shields.io/github/forks/Radical3-HeZhang/Awesome-Trustworthy-GNNs?color=blue&label=Fork)


This is a collection of resources related to trustworthy graph neural networks.

## Contents

- [Related concepts](#concepts)
- [Papers](#papers)
  - [Robustness](#robustness)
	  - [Attacks](#robustness-attack)
	  - [Defences](#robustness-defence)
  - [Explainability](#explainability)
	  - [Interpretable GNNs](#explainability-self)
	  - [Post-hoc Explainers](#explainability-post)
  - [Privacy](#privacy) 
	  - [Privacy Attacks](#privacy-attack)
	  - [Privacy-preserving Techniques for GNNs](#privacy-preserving)
  - [Fairness](#fairness)
  - [Accountability](#accountability)
  - [Environmental well-being](#env)
	  - [Scalable GNN Architectures and Efficient Data Communication](#env-scale)
	  - [Model Compression Methods](#env-compression)
	  - [Efficient Frameworks and Accelerators](#env-swhw)
  - [Others](#others)
  - [Relations](#relations)

<a name="concepts" />

## Related concepts

### Trustworthy GNNs
1. **Trustworthy Graph Neural Networks: Aspects, Methods and Trends.** *He Zhang, Bang Wu, Xingliang Yuan, Shirui Pan, Hanghang Tong, Jian Pei.* 2022. [paper](https://arxiv.org/abs/2205.07424)
2. **A Comprehensive Survey on Trustworthy Graph Neural Networks: Privacy, Robustness, Fairness, and Explainability.** *Enyan Dai, Tianxiang Zhao, Huaisheng Zhu, Junjie Xu, Zhimeng Guo, Hui Liu, Jiliang Tang, Suhang Wang.* 2022. [paper](https://arxiv.org/abs/2204.08570)

### Graph Neural Networks
1. **A Comprehensive Survey on Graph Neural Networks.** *Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, Philip S. Yu.* 2019. [paper](https://ieeexplore.ieee.org/document/9046288)
2. **Graph Neural Networks: Foundations, Frontiers, and Applications.** *Lingfei Wu, Peng Cui, Jian Pei, Liang Zhao.* 2022. [book](https://graph-neural-networks.github.io/index.html)

### Trustworthy AI / ML
1. **Trustworthy AI: A computational perspective.** *Haochen Liu, Yiqi Wang, Wenqi Fan, Xiaorui Liu, Yaxin Li, Shaili Jain, Yunhao Liu, Anil K. Jain, Jiliang Tang.* 2021. [paper](https://arxiv.org/pdf/2107.06641.pdf)
2. **Trustworthy AI: from principles to practices.** *Bo Li], Peng Qi, Bo Liu, Shuai Di, Jingen Liu, Jiquan Pei, Jinfeng Yi, Bowen Zhou.* 2021 [paper](https://arxiv.org/pdf/2110.01167.pdf)
3. **Trustworthy Machine Learning.** *Kush R. Varshney.* 2022. [book](http://www.trustworthymachinelearning.com/)

<a name="papers" />

## Papers

Here we only list some papers. For other studies, please visit our [Survey on Trustworthy GNNs](https://arxiv.org/abs/2205.07424). 

<a name="robustness" />

## Robustness

<a name="robustness-attack" />

### Attacks

1. **Adversarial attack on graph structured data.** ICML 2018. [paper](http://proceedings.mlr.press/v80/dai18b/dai18b.pdf)
2. **Topology attack and defense for graph neural networks: An optimization perspective.** IJCAI 2019. [paper](https://www.ijcai.org/Proceedings/2019/0550.pdf)
3. **Adversarial examples for graph data: Deep insights into attack and
defense.** IJCAI 2019. [paper](https://www.ijcai.org/proceedings/2019/0669.pdf)
4.  **Fast gradient attack on network embedding.** Arxiv 2018. [paper](https://arxiv.org/pdf/1809.02797.pdf)
5. **Derivative-free optimization adversarial attacks for graph convolutional networks.** PeerJ Computer Science 2021. [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8409335/pdf/peerj-cs-07-693.pdf)
6. **Adversarial attacks on graph neural networks via node injections: A hierarchical reinforcement learning approach.** WWW 2020. [paper](https://dl.acm.org/doi/10.1145/3366423.3380149)
7. **Adversarial attacks on neural networks for graph data.** KDD 2018. [paper](https://dl.acm.org/doi/abs/10.1145/3219819.3220078)

<a name="robustness-defence" />

### Defences

1. **All you need is low (rank): Defending against adversarial attacks on graphs.** WSDM 2020. [paper](https://dl.acm.org/doi/10.1145/3336191.3371789)
2. **Graph structure learning for robust graph neural networks.** KDD 2020. [paper](https://dl.acm.org/doi/abs/10.1145/3394486.3403049)
3. **Graph sanitation with application to node classification.** WWW 2022. [paper](https://dl.acm.org/doi/abs/10.1145/3485447.3512180)
4. **Robust graph convolutional networks against adversarial attacks.** KDD 2019. [paper](https://dl.acm.org/doi/abs/10.1145/3292500.3330851)
5. **Transferring robustness for graph neural network against poisoning attacks.** WSDM 2020. [paper](https://dl.acm.org/doi/abs/10.1145/3336191.3371851)
6. **Defending graph convolutional networks against adversarial attacks.** IEEE ICASSP 2020. [paper](https://ieeexplore.ieee.org/abstract/document/9054325)
7. **Gnnguard: Defending graph neural networks against adversarial attacks.** NeurIPS 2020. [paper](https://proceedings.neurips.cc/paper/2020/file/690d83983a63aa1818423fd6edd3bfdb-Paper.pdf)
8. **Graph adversarial training: Dynamically regularizing based on graph structure.** IEEE TKDE 2021. [paper](https://ieeexplore.ieee.org/abstract/document/8924766)
9. **Robust training of graph convolutional networks via latent perturbation.** PKDD 2020. [paper](https://link.springer.com/chapter/10.1007/978-3-030-67664-3_24)
10. **Topology attack and defense for graph neural networks: An optimization perspective.** IJCAI 2019. [paper](https://www.ijcai.org/Proceedings/2019/0550.pdf)
11. **Certifiable robustness to graph perturbations.** NeurIPS 2019. [paper](https://proceedings.neurips.cc/paper/2019/file/e2f374c3418c50bc30d67d5f7454a5b4-Paper.pdf)
12. **Certifiable robustness and robust training for graph convolutional networks.** KDD 2019. [paper](https://dl.acm.org/doi/abs/10.1145/3292500.3330905)
13. **Adversarial immunization for certifiable robustness on graphs.** WSDM 2021. [paper](https://dl.acm.org/doi/abs/10.1145/3437963.3441782)
14. **Comparing and detecting adversarial attacks for graph deep learning.** ICLR 2019. [paper](https://rlgm.github.io/papers/57.pdf)

<a name="explainability" />

## Explainability

<a name="explainability-self" />

### Interpretable GNNs
1. **Convolutional networks on graphs for learning molecular fingerprints.** NeurIPS 2015. [paper](https://papers.nips.cc/paper/2015/file/f9be311e65d81a9ad8150a60844bb94c-Paper.pdf)
2. **Substructure assembling network for graph classification.** AAAI 2018. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/11742/11601)
3. **Towards explainable representation of time-evolving graphs via spatial-temporal graph attention networks.** CIKM 2019. [paper](https://dl.acm.org/doi/10.1145/3357384.3358155)
4. **Towards self-explainable graph neural network.** CIKM 2021. [paper](https://dl.acm.org/doi/10.1145/3459637.3482306)
5. **Protgnn: Towards self-explaining graph neural networks.** AAAI 2022. [paper](https://arxiv.org/pdf/2112.00911.pdf)
6. **Motif-driven contrastive learning of graph representations.** AAAI 2021. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/17986)
7. **Discovering invariant rationales for graph neural networks.** ICLR 2022. [paper](https://openreview.net/pdf?id=hGXij5rfiHw)
8. **Graph information bottleneck for subgraph recognition.** ICLR 2021. [paper](https://openreview.net/pdf?id=bM4Iqfg8M2k)

<a name="explainability-post" />

### Post-hoc Explainers
1. **Explainability techniques for graph convolutional networks.** ICML 2019. [paper](https://arxiv.org/pdf/1905.13686.pdf)
2. **Explainability methods for graph convolutional neural networks.** CVPR 2019. [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pope_Explainability_Methods_for_Graph_Convolutional_Neural_Networks_CVPR_2019_paper.pdf)
3. **Gnnexplainer: Generating explanations for graph neural networks.** NeurIPS 2019. [paper](https://cs.stanford.edu/people/jure/pubs/gnnexplainer-neurips19.pdf)
4. **Parameterized explainer for graph neural network.** NeurIPS 2020. [paper](https://dl.acm.org/doi/pdf/10.5555/3495724.3497370)
5. **Hard masking for explaining graph neural networks.** OpenReview 2021. [paper](https://openreview.net/forum?id=uDN8pRAdsoC) 
6. **Causal screening to interpret graph neural networks.** OpenReview 2020. [paper](https://openreview.net/forum?id=nzKv5vxZfge)
7. **Interpreting graph neural networks for NLP with differentiable edge masking.** ICLR 2021. [paper](https://openreview.net/pdf?id=WznmQa42ZAx)
8. **On explainability of graph neural networks via subgraph explorations.** ICML 2021. [paper](http://proceedings.mlr.press/v139/yuan21c/yuan21c.pdf)
9. **Cf-gnnexplainer: Counterfactual explanations for graph neural networks.** AISTATS 2022. [paper](https://proceedings.mlr.press/v151/lucic22a/lucic22a.pdf)
10. **Robust counterfactual explanations on graph neural networks.** NeurIPS 2021. [paper](https://openreview.net/pdf?id=wGmOLwb8ClT) 
11. **Towards multi-grained explainability for graph neural networks.** NeurIPS 2021. [paper](https://openreview.net/pdf?id=e5vrkfc5aau)
12. **Learning and evaluating graph neural network explanations based on counterfactual and factual reasoning.** WWW 2022. [paper](https://dl.acm.org/doi/pdf/10.1145/3485447.3511948)
13. **Graphlime: Local interpretable model explanations for graph neural networks.** Arxiv 2020. [paper](https://arxiv.org/pdf/2001.06216.pdf)
14. **Relex: A model-agnostic relational model explainer.** AIES 2021. [paper](https://dl.acm.org/doi/pdf/10.1145/3461702.3462562)
15. **Pgm-explainer: Probabilistic graphical model explanations for graph neural networks.** NeurIPS 2020. [paper](https://proceedings.neurips.cc/paper/2020/file/8fb134f258b1f7865a6ab2d935a897c9-Paper.pdf)
16. **Higher-order explanations of graph neural networks via relevant walks.** TPAMI 2021. [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9547794)
17. **XGNN: towards model-level explanations of graph neural networks.** KDD 2020. [paper](https://dl.acm.org/doi/pdf/10.1145/3394486.3403085)
18. **Reinforcement learning enhanced explainer for graph neural networks.** NeurIPS 2021. [paper](https://proceedings.neurips.cc/paper/2021/file/be26abe76fb5c8a4921cf9d3e865b454-Paper.pdf)
19. **Orphicx: A causality-inspired latent variable model for interpreting graph neural networks.** CVPR 2022. [paper](https://wanyu-lin.github.io/assets/publications/wanyu-cvpr2022.pdf)
20. **DEGREE: Decomposition based explanation for graph neural networks.** ICLR 2021. [paper](https://openreview.net/pdf?id=Ve0Wth3ptT_)
21. **Counterfactual graphs for explainable classification of brain networks.** KDD 2021. [paper](https://dl.acm.org/doi/pdf/10.1145/3447548.3467154)
22. **Generative causal explanations for graph neural networks.** ICML 2021. [paper](http://proceedings.mlr.press/v139/lin21d/lin21d.pdf)

<a name="privacy" />

## Privacy

<a name="privacy-attack" />

### Privacy Attacks

1. **Model extraction attacks on graph neural networks: Taxonomy and realization.** AsiaCCS 2022. [paper](https://arxiv.org/pdf/2010.12751.pdf)
2. **Learning discrete structures for graph neural networks.** ICML 2019. [paper](http://proceedings.mlr.press/v97/franceschi19a.html)
3. **Quantifying privacy leakage in graph embedding.** MobiQuitous 2020. [paper](https://dl.acm.org/doi/abs/10.1145/3448891.3448939)
4. **Node-level membership inference attacks against graph neural networks.** Arxiv 2021. [paper](https://arxiv.org/pdf/2102.05429.pdf)
5. **Stealing links from graph neural networks.** USENIX Security Symposium 2021. [paper](https://www.usenix.org/system/files/sec21-he-xinlei.pdf)
6. **Adapting membership inference attacks to GNN for graph classification: Approaches and implications.** IEEE ICDM 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9679062)
7. **Membership inference attacks on knowledge graphs.** Arxiv 2021. [paper](https://arxiv.org/pdf/2104.08273.pdf)
8. **Inference attacks against graph neural networks.** USENIX Security Symposium 2022. [paper](https://www.usenix.org/system/files/sec22summer_zhang-zhikun.pdf)
9. **Graphmi: Extracting private graph data from graph neural networks.** IJCAI 2021. [paper](https://www.ijcai.org/proceedings/2021/0516.pdf)
10. **Linkteller: Recovering private edges from graph neural networks via influence analysis.** IEEE S&P 2022. [paper](https://par.nsf.gov/servlets/purl/10289325) 
11.  **Privacy-preserving representation learning on graphs: A mutual information perspective.** KDD 2021. [paper](https://dl.acm.org/doi/abs/10.1145/3447548.3467273)

<a name="privacy-preserving" />

### Privacy-preserving Techniques for GNNs

<a name="privacy-preserving-FL" />

#### Federated Learning

1. **Federated dynamic graph neural networks with secure aggregation for video-based distributedsurveillance.** IEEE TIST 2022. [paper](https://dl.acm.org/doi/10.1145/3501808)
2. **Spreadgnn: Serverless multi-task federated learning for graph neural networks.** Arxiv 2021. [paper](https://arxiv.org/pdf/2106.02743.pdf)
3. **Federated graph classification over non-iid graphs.** NeurIPS. [paper](https://proceedings.neurips.cc/paper/2021/file/9c6947bd95ae487c81d4e19d3ed8cd6f-Paper.pdf)
4. **A federated multigraph integration approach for connectional brain template learning.** ML-CDS 2021. [paper](https://link.springer.com/chapter/10.1007/978-3-030-89847-2_4)
5. **Federated learning of molecular properties in a heterogeneous setting.** Arxiv 2021. [paper](https://arxiv.org/pdf/2109.07258.pdf)
6. **STFL: A temporal-spatial federated learning framework for graph neural networks.** Arxiv 2021. [paper](https://arxiv.org/pdf/2111.06750.pdf)
7. **Fedgnn: Federated graph neural network for privacy-preserving recommendation.** Arxiv 2021. [paper](https://arxiv.org/pdf/2102.04925.pdf)
8. **Federated social recommendation with graph neural network.** Arxiv 2021. [paper](https://arxiv.org/abs/2111.10778)
9. **A vertical federated learning framework for graph convolutional network.** Arxiv 2021. [paper](https://arxiv.org/pdf/2106.11593.pdf)
10. **Vertically federated graph neural network for privacypreserving node classification.** Arxiv 2020. [paper](https://arxiv.org/pdf/2005.11903.pdf)
11. **ASFGNN: automated separated-federated graph neural network.** Peer-to-Peer Networking and Applications 2021. [paper](https://link.springer.com/article/10.1007/s12083-021-01074-w)
12. **Graphfl: A federated learning framework for semi-supervised node classification on graphs.** Arxiv 2020. [paper](https://arxiv.org/pdf/2012.04187.pdf)
13. **Fedgl: Federated graph learning framework with global self-supervision.** Arxiv 2021. [paper](https://arxiv.org/pdf/2105.03170.pdf)
14. **Cross-node federated graph neural network for spatio-temporal data modeling.** KDD 2021. [paper](https://dl.acm.org/doi/abs/10.1145/3447548.3467371)
15. **Subgraph federated learning with missing neighbor generation.** NeurIPS 2021. [paper](https://proceedings.neurips.cc/paper/2021/file/34adeb8e3242824038aa65460a47c29e-Paper.pdf)
16. **Fedgraph: Federated graph learning with intelligent sampling.** IEEE TPDS 2022. [paper](https://ieeexplore.ieee.org/abstract/document/9606516)
17. **Towards representation identical privacy-preserving graph neural network via split learning.** Arxiv 2021. [paper](https://arxiv.org/pdf/2107.05917.pdf)
18. **Fedgraphnn: A federated learning system and benchmark for graph neural networks.** Arxiv 2021. [paper](https://arxiv.org/pdf/2104.07145.pdf) 

<a name="privacy-preserving-DP" />

#### Differential Privacy

1. **Locally private graph neural networks.** ACM CCS 2021. [paper](https://dl.acm.org/doi/10.1145/3460120.3484565)
2. **Graph embedding for recommendation against attribute inference attacks.** WWW 2021. [paper](https://dl.acm.org/doi/abs/10.1145/3442381.3449813)

<a name="privacy-preserving-IT" />

#### Insusceptible Training

1. **Netfense: Adversarial defenses against privacy attacks on neural networks for graph data.** IEEE TKDE 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9448513)
2. **Information obfuscation of graph neural networks.** ICML 2021. [paper](http://proceedings.mlr.press/v139/liao21a/liao21a.pdf)
3. **Adversarial privacypreserving graph embedding against inference attack.** IEEE ITJ 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9250489)

<a name="fairness" />

## Fairness
1. **Compositional fairness constraints for graph embeddings.** ICML 2019. [paper](http://proceedings.mlr.press/v97/bose19a/bose19a.pdf)
2. **Say no to the discrimination: Learning fair graph neural networks with limited sensitive attribute information.** WSDM 2021. [paper](https://dl.acm.org/doi/pdf/10.1145/3437963.3441752)
3. **Towards a unified framework for fair and stable graph representation learning.** UAI 2021. [paper](https://proceedings.mlr.press/v161/agarwal21b/agarwal21b.pdf)
4. **EDITS: modeling and mitigating data bias for graph neural networks.** WWW 2022. [paper](https://dl.acm.org/doi/pdf/10.1145/3485447.3512173)
5. **Inform: Individual fairness on graph mining.** KDD 2020. [paper](https://dl.acm.org/doi/pdf/10.1145/3394486.3403080)
6. **On dyadic fairness: Exploring and mitigating bias in graph connections.** ICLR 2021. [paper](https://openreview.net/pdf?id=xgGS6PmzNq6)
7. **Fairdrop: Biased edge dropout for enhancing fairness in graph representation learning.** IEEE TAI 2021. [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9645324)
8. **Individual fairness for graph neural networks: A ranking based approach.** KDD 2021. [paper](https://dl.acm.org/doi/pdf/10.1145/3447548.3467266)

<a name="accountability" />

## Accountability
1. **A pipeline for fair comparison of graph neural networks in node classification tasks.** Arxiv 2020. [paper](https://arxiv.org/pdf/2012.10619.pdf)
2. **A novel genetic algorithm with hierarchical evaluation strategy for hyperparameter optimisation of graph neural networks.** Arxiv 2021. [paper](https://arxiv.org/pdf/2101.09300.pdf)
3. **Bag of tricks of semi-supervised classification with graph neural networks.** Arxiv 2021. [paper](https://arxiv.org/pdf/2103.13355v4.pdf)
4. **Bag of tricks for training deeper graph neural networks: A comprehensive benchmark study.** IEEE TPAMI 2022. [paper](https://ieeexplore.ieee.org/abstract/document/9773017)
5. **A pipeline for fair comparison of graph neural networks in node classification tasks.** Arxiv 2020. [paper](https://arxiv.org/pdf/2012.10619.pdf)
6. **A fair comparison of graph neural networks for graph classification.** Arxiv 2019. [paper](https://arxiv.org/pdf/1912.09893.pdf)
7. **HASHTAG: hash signatures for online detection of fault-injection attacks on deep neural networks.** ICCAD 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9643556)
8. **Sensitive-sample fingerprinting of deep neural networks.** CVPR 2019. [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Sensitive-Sample_Fingerprinting_of_Deep_Neural_Networks_CVPR_2019_paper.pdf)
9. **Proof-of-learning: Definitions and practice.** IEEE S&P 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9519402)
10. **Proof of learning (pole): Empowering machine learning with consensus building on blockchains (demo).** AAAI 2021. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/18013)

<a name="env" />

## Environmental well-being

<a name="env-scale" />

### Scalable GNN Architectures and Efficient Data Communication
1. **GraphSAINT: Graph Sampling Based Inductive Learning Method.** ICLR 2020. [paper](https://openreview.net/pdf?id=BJe8pkHFwS)
2. **Gnnautoscale: Scalable and expressive graph neural networks via historical embeddings.** ICML 2021. [paper](http://proceedings.mlr.press/v139/fey21a/fey21a.pdf)
3. **Simplifying graph convolutional networks.** ICML 2019. [paper](http://proceedings.mlr.press/v97/wu19e/wu19e.pdf)
4. **Training graph neural networks with 1000 layers.** ICML 2021. [paper](http://proceedings.mlr.press/v139/li21o/li21o.pdf)
5. **Pinnersage: Multi-modal user embedding framework for recommendations at pinterest.** KDD 2020. [paper](https://cs.stanford.edu/people/jure/pubs/pinnersage-kdd20.pdf)
6. **ETA prediction with graph neural networks in google maps.** CIKM 2021. [paper](https://dl.acm.org/doi/abs/10.1145/3459637.3481916)
7. **# Efficient Data Loader for Fast Sampling-Based GNN Training on Large Graphs.** IEEE TPDS 2021. [paper](https://ieeexplore.ieee.org/document/9376972)

<a name="env-compression" />

### Model Compression Methods
1. **On self-distilling graph neural network.** IJCAI 2021. [paper](https://www.ijcai.org/proceedings/2021/0314.pdf)
2. **Graph-free knowledge distillation for graph neural networks.** IJCAI 2021. [paper](https://www.ijcai.org/proceedings/2021/0320.pdf)
3. **Tinygnn: Learning efficient graph neural networks.** KDD 2020. [paper](https://dl.acm.org/doi/abs/10.1145/3394486.3403236)
4. **A unified lottery ticket hypothesis for graph neural networks.** ICML 2021. [paper](http://proceedings.mlr.press/v139/chen21p/chen21p.pdf)
5. **Graph normalizing flows.** NeurIPS 2019. [paper](https://proceedings.neurips.cc/paper/2019/file/1e44fdf9c44d7328fecc02d677ed704d-Paper.pdf)
6. **Binary graph neural networks.** CVPR 2021. [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Bahri_Binary_Graph_Neural_Networks_CVPR_2021_paper.pdf)
7. **Degree-quant: Quantization-aware training for graph neural networks.** ICLR 2021. [paper](https://openreview.net/pdf?id=NSBrFgJAHg)

<a name="env-swhw" />

### Efficient Frameworks and Accelerators
1. **Fast graph representation learning with PyTorch Geometric.** ICLR 2019. [paper](https://rlgm.github.io/papers/2.pdf)
2. **Deep graph library: Towards efficient and scalable deep learning on graphs.** ICLR 2019. [paper](https://rlgm.github.io/papers/49.pdf)
3. **Engn: A high-throughput and energy-efficient accelerator for large graph neural networks.** IEEE TC 2021. [paper](https://ieeexplore.ieee.org/abstract/document/9161360)
4. **Hygcn: A GCN accelerator with hybrid architecture.** HPCA 2020. [paper](https://par.nsf.gov/servlets/purl/10188415)
5. **Characterizing and understanding gcns on GPU.** IEEE CAL. [paper](https://ieeexplore.ieee.org/abstract/document/8976117)
6. **Alleviating irregularity in graph analytics acceleration: a hardware/software co-design approach.** MICRO 2019. [paper](https://miglopst.github.io/files/yan_micro2019.pdf)
7. **Accelerating large scale real-time GNN inference using channel pruning.** VLDB Endowment 2021. [paper](https://arxiv.org/pdf/2105.04528.pdf)
8. **G-cos: Gnnaccelerator co-search towards both better accuracy and efficiency.** IEEE ICCAD. [paper]()

<a name="others" />

## Others
1. **How neural networks extrapolate: From feedforward to graph neural networks.** ICLR 2021. [paper](https://openreview.net/forum?id=UH-cmocLJC)

<a name="relations" />

## Relations

1. **Explainability-based backdoor attacks against graph neural networks.** WiseML 2021. [paper](https://dl.acm.org/doi/abs/10.1145/3468218.3469046)
2. **Jointly attacking graph neural network and its explanations.** Arxiv 2021. [paper](https://arxiv.org/pdf/2108.03388.pdf)
3. **Towards a unified framework for fair and stable graph representation learning.** UAI 2021. [paper](https://proceedings.mlr.press/v161/agarwal21b/agarwal21b.pdf)
4. **Compositional fairness constraints for graph embeddings.** ICML 2019. [paper](http://proceedings.mlr.press/v97/bose19a/bose19a.pdf)
5.  **Say no to the discrimination: Learning fair graph neural networks with limited sensitive attribute information.** WSDM 2021. [paper](https://dl.acm.org/doi/pdf/10.1145/3437963.3441752)
6. **Discrete-valued neural communication.** NeurIPS 2021. [paper](https://papers.nips.cc/paper/2021/file/10907813b97e249163587e6246612e21-Paper.pdf) 
7. **Graph structure learning for robust graph neural networks.** KDD 2020. [paper](https://dl.acm.org/doi/10.1145/3394486.3403049) 
8. **Defensevgae: Defending against adversarial attacks on graph data via a variational graph autoencoder.** Arxiv 2020. [paper](https://arxiv.org/pdf/2006.08900.pdf) 
9. **Robust graph convolutional networks against adversarial attacks.** KDD 2019. [paper](https://dl.acm.org/doi/abs/10.1145/3292500.3330851) 
10. **Transferring robustness for graph neural network against poisoning attacks.** WSDM 2020. [paper](https://dl.acm.org/doi/abs/10.1145/3336191.3371851) 
11. **Extract the knowledge of graph neural networks and go beyond it: An effective knowledge distillation framework.** WWW 2021. [paper](https://dl.acm.org/doi/abs/10.1145/3442381.3450068) 
12. **Privacy-preserving representation learning on graphs: A mutual information perspective.** KDD 2021. [paper](https://dl.acm.org/doi/abs/10.1145/3447548.3467273) 
13. **Topological uncertainty: Monitoring trained neural networks through persistence of activation graphs.** IJCAI 2021. [paper](https://www.ijcai.org/proceedings/2021/0367.pdf)  




If you need more details, please visit the [Survey on Trustworthy GNNs](https://arxiv.org/abs/2205.07424).
```
@article{DBLP:journals/corr/abs-2205-07424,
  author    = {He Zhang and
               Bang Wu and
               Xingliang Yuan and
               Shirui Pan and
               Hanghang Tong and
               Jian Pei},
  title     = {Trustworthy Graph Neural Networks: Aspects, Methods and Trends},
  journal   = {CoRR},
  volume    = {abs/2205.07424},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2205.07424},
  doi       = {10.48550/arXiv.2205.07424},
  eprinttype = {arXiv},
  eprint    = {2205.07424}
}
```

