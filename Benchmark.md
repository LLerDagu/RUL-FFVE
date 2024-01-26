## Benchmark
- To facilitate subsequent research, we have provided extensive comparative results for various methods.
- If there are any bugs or paper errata, we will promptly update them here!

Note: 
Due to the inability to use the equipment mentioned in the paper, we used a new device to compute the results.
Therefore, there may be some variations in SCORE compared to the original results (SCORE will remain unchanged if the same equipment is used). 
If you need to make performance comparisons, you can refer to these new results.

Furthermore, from the experimental results, all the conclusions mentioned in the paper still hold true, so there is no need to worry about any textual errors in the paper.

## Table 4
Comparison of the proposed method FFVE with SOTA methods on the C-MAPSS turbofan engine dataset in terms of RMSE and SCORE, where lower values indicating better performance. The best results are highlighted in **bold** and the second-best results are in _italics_. The IMP shows the improvement of our method over the best result of the SOTA methods.

| Type                  | Method         | FD001      |             | FD002       |             | FD003      |             | FD004       |              |
| :-------------------- | :------------- | :---------: | :----------: | :----------: | :----------: | :---------: | :----------: | :----------: | :-----------: |
|                       |                | RMSE       | SCORE       | RMSE        | SCORE       | RMSE       | SCORE       | RMSE        | SCORE        |
| Traditional ML        | MLP            | 37.56     | 18000      | 80.03      | 7800000   | 37.39     | 17400      | 77.37      | 5620000    |
|                       | SVR            | 20.96     | 1380        | 42.00      | 590000     | 21.05     | 1600        | 45.35      | 371000      |
|                       | RVR            | 23.80     | 1500        | 31.30      | 17400      | 22.37     | 1430        | 34.34      | 26500       |
| Deep Learning         | CNN            | 18.45     | 1299        | 30.29      | 13600      | 13 600     | 1600        | 29.16      | 7890         |
|                       | Deep LSTM      | 16.14     | 338         | 24.49      | 4450        | 16.18     | 852         | 28.17      | 5550         |
|                       | AGCNN          | 12.42     | 225.5      | 19.43      | 1492        | 13.39     | 227.1      | 21.50      | 3392         |
|                       | MS-DCNN        | _11.44_     | 196.22     | 19.35      | 3747        | _11.67_     | 241.89     | 22.22      | 4844         |
|                       | Bi-level LSTM  | 11.80     | _194_         | 23.14      | 3771        | 12.37     | _224_         | 23.38      | 3492         |
|                       | ADLDNN         | 13.05     | 238         | 17.33      | _1149_        | 12.59     | 281         | 16.95      | 1371         |
|                       | MLE(4X)+CCF    | 11.57     | 208         | 18.84      | 1606        | 11.83     | 262         | 20.78      | 2081         |
|                       | MTSTAN         | **10.97** | **175.36** | 16.81      | 1154.36    | **10.90** | **188.22** | 18.85      | 1446.29     |
| Interpretable methods | TaFCN          | 13.99     | 336         | 19.59      | 2650        | 19.16     | 1727        | 22.15      | 2901         |
|                       | Standard VAE   | 13.74     | 339         | 17.91      | 1543        | 14.52     | 436         | 19.27      | 1987         |
|                       | RVE            | 13.42     | 323.82     | 14.92      | 1379.17    | 12.51     | 256.36     | _16.37_      | 1845.99     |
|                       | TF-SCN+HLS-VAE | 12.05     | 219         | _14.71_      | 1358        | 12.11     | 238         | 16.95      | _1367_         |
|                       | **FFVE**       | 12.31     | 280.21     | **12.55**  | **772.17** | 12.43     | 302.1      | **14.08**  | **1149.03** |
| **IMP**               |                | -          | -           | **14.68%** | **32.80%** | -          | -           | **13.99%** | **15.95%**  |


## Table 7
Comparison of the proposed method FFVE with different numbers of FTCF blocks and feature fusion types.
- The X1 means FFVE with one FTCF block, and so on for X2, X3.
- <sup>a</sup> The selected architecture on all four datasets.

| The number Of FTCF Block | Feature fusion operation | FD001 |         | FD002 |         | FD003 |         | FD004 |          |
|:------------------------:|:------------------------:|:-----:|:-------:|:-----:|:-------:|:-----:|:-------:|:-----:|:--------:|
|                          |                          | RMSE  | SCORE   | RMSE  | SCORE   | RMSE  | SCORE   | RMSE  | SCORE    |
| X1                       | short range              | **12.31**<sup>a</sup> | **280.21**<sup>a</sup>  | 12.92 | **767.27**  | **12.43**<sup>a</sup> | **302.1**<sup>a</sup>   | 14.27 | 1233.02  |
|                          | long-short range         | 12.71 | _285.13_  | 12.92 | 778.59  | 12.85 | 354     | 14.17 | **1141.55**  |
| X2                       | short range              | 12.73 | 322.34  | **12.55**<sup>a</sup> | _772.17_<sup>a</sup>  | _12.6_  | 366.76  | 14.27 | 1355.74  |
|                          | long-short range         | _12.35_ | 298.28  | 12.81 | 801.7   | 12.64 | _349.94_  | _14.08_<sup>a</sup> | _1149.03_<sup>a</sup>  |
| X3                       | short range              | 12.85 | 333.31  | _12.78_ | 777.22  | 12.97 | 374.9   | 14.34 | 1457.84  |
|                          | long-short range         | 12.93 | 337.4   | 12.93 | 819.58  | 12.72 | 403.11  | **14.02** | 1225.26  |

## Table 8
Summary of average RMSE/SCORE for FFVE when excluding three types of modules.

| Number | Method                    | Factorized <br> Channel | Feature <br> Fusion | 3D Latent <br> Space | FD001                   | FD002               | FD003               | FD004                    |
| :-----: | :------------------------ | :---------------------: | :-----------------: | :------------------: | :----------------------: | :------------------: | :------------------: | :-----------------------: |
| (1)    | FFVE  w/o factorization   |                         | ✔                   | ✔                    | _12.42_ / 307.79         | 13.68 / 943.61     | 13.31 / 381.78     | 16.40 / 1555.22         |
| (2)    | FFVE w/ channel attention |                         | ✔                   | ✔                    | 12.60 / 297.78         | 13.05 / _723.65_     | 12.57 / **300.81** | 14.58 / 1899            |
| (3)    | FFVE  w/o feature fusion  | ✔                       |                     | ✔                    | 12.73 / 324.89         | 43.47 / 62588.93   | 14.14 / 389.15     | 44.27 / 105934.16       |
| (4)    | FFVE  w/o 3D              | ✔                       | ✔                   |                      | 12.48 / _291.2_          | _12.62_ / **689.43** | _12.49_ / 305.27     | _14.1_ / _1166.48_          |
| (5)    | FFVE                      | ✔                       | ✔                   | ✔                    | **12.31** / **280.21** | **12.55** / 772.17 | **12.43** / _302.1_  | **14.08** / **1149.03** |
