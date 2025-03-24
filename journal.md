Week of 3/3:
- Did a little bit of basic EDA (following the docs online) on the Weave dataset -- it is not feasible to download the entire dataset. Rather you apply filters to get data from specific intervals. I think I'm going to have to make a script that does this across many such intervals and then downloads them? That way I dont have to keep pulling from s3 into memory.
- Looked a little bit at GNNs and I am moving towards exploring the following models:
    - Separate Spatial and Temporal Components:
        - I want to explore 3 different architectures for the temporal component (LSTMs, TCNs, and Transformers)
        - I will likely use a Graph Attention Netwrok or Graph Convolution Network for the Spatial Component
        - I think a study of different combinations of these networks (6 total) will make for good analysis 
    - Single Sppatiotemporal Graph Convolutional Network (ST-GCN):
        - I will extend GCN by adding temporal convolutions, processing time-series data in parallel rather than sequentially
        - I hope to compare this to two-component architectures

Week of 3/10:
- Explored the Weave Dataset a lot more, thinking primarily about how I want to tackle a dataset of this size
     - Browsed through some of the specific parquet files in the Weave S3 bucket -- it appears they've split the data by month. 
     - Perhaps, the best course of action is to zero in on a specific month and download the corresponding parquet file. I think this avoids the seasonal changes that occur which may likely be an exogenous variable affecting energy consumption

Week of 3/17:
- Explored this repo (https://github.com/ddz16/TSFpaper?tab=readme-ov-file#tcncnn) which provided links to papers, some of which were on spatio-temporal forecasting
- Did a bit of general reading on graph neural networks:
    - Read this gentle introduction: https://distill.pub/2021/gnn-intro/
    - Read the Graph Attention Networks Paper: https://arxiv.org/abs/1710.10903
