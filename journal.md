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
- 