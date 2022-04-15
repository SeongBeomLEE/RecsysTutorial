# RecsysTutorial
추천시스템 논문을 읽고 구현한 Code가 저장된 Recsys Tutorial Repository 입니다.

논문 리뷰가 저장된 벨로그 주소: https://velog.io/@2712qwer/series/RecSys-Paper

## Used Data
- [MovieLens - 100k](https://www.kaggle.com/rajmehra03/movielens100k)
- [Criteo CTR DataSet - 1GB](https://www.kaggle.com/c/mlbd-20-ctr-prediction-1/data)

## Implementation List

### Session-based

- [GRU4Rec]() - MovieLens
  - (2016). Session-based Recommendations with Recurrent Neural Networks

- [Caser](https://github.com/SeongBeomLEE/RecsysTutorial/tree/main/Caser) - MovieLens
  - (2018). Personalized Top-N Sequential Recommendation Via Convolutional Sequence Embedding

- [SASRec]() - MovieLens
  - (2018). Self-Attentive Sequential Recommendation

- [BERT4Rec]() - MovieLens
  - (2019). BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer

- S3-Rec - MovieLens
  - (2020). S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization

### GNN-based

- SR-GNN - MovieLens
  - (2018). Session-based Recommendation with Graph Neural Networks

- [NGCF]() - MovieLens
  - (2019), Neural Graph Collaborative Filtering

- [LightGCN]() - MovieLens
  - (2020), LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
  
- GC-MC - MovieLens
  - (2017). Graph Convolutional Matrix Completion
  
- PinSAGE - MovieLens
  - (2018). Graph Convolutional Neural Networks for Web-Scale Recommender Systems

### AutoEncoder-based

- [AutoRec](https://github.com/SeongBeomLEE/RecsysTutorial/tree/main/AutoRec) - MovieLens
  - (2015). Autorec: Autoencoders Meet Collaborative Filtering

- [Multi-VAE & Multi-DAE](https://github.com/SeongBeomLEE/RecsysTutorial/tree/main/Multi-VAE-and-Multi-DAE) - MovieLens
  - (2018). Variational Autoencoders for Collaborative Filtering

- [RecVAE]() - MovieLens
  - (2019). RecVAE: a New Variational Autoencoder for Top-N Recommendations with Implicit Feedback

- [EASE]() - MovieLens
  - (2019). Embarrassingly Shallow Autoencoders for Sparse Data
  
- [ADMM SLIM]() - MovieLens
  - (2020). ADMM SLIM: Sparse Recommendations for Many Users

- [EASER]() - MovieLens
  - (2021). Negative Interactions for Improved Collaborative Filtering: Don’t go Deeper, go Higher

### 이외

- [MF](https://github.com/SeongBeomLEE/RecsysTutorial/tree/main/MF) - MovieLens
  - (2009). Matrix Factorization Techniques for Recommender Systems
  
- [NCF](https://github.com/SeongBeomLEE/RecsysTutorial/tree/main/NCF) - MovieLens
  - (2017). Neural Collaborative Filtering
  
- [BPR](https://github.com/SeongBeomLEE/RecsysTutorial/tree/main/BPR) - MovieLens
  - (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback

- [FM]() - CTR
  - (2010). Factorization Machines
  
- [DeepFM]() - CTR
  - (2017). DeepFM: A Factorization-Machine based Neural Network for CTR Prediction

- [Item2Vec]() - MovieLens
  - (2016). Item2Vec : Neural Item Embedding for Collaborative Filtering
