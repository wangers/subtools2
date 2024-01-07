## Reports

### Results of CamPPlus (2024-01-07)

The configuration is based on [3D-Speaker](https://github.com/alibaba-damo-academy/3D-Speaker/tree/main/egs/voxceleb/sv-cam%2B%2B), originally released in the [CAM++ paper](https://arxiv.org/abs/2303.00332), with the exception of the sampling strategy switch (from sequential chunk to random chunk).
* Egs = Voxceleb2_dev(online random aug) + random chunk(**2s** or **3s**)
* Optimization = [sgd (lr = 0.1 - 1e-5) + warm_cosine] x 4 GPUs (total batch-size=1024) + 120 epochs + average best 5
* CamPPlus + AAM-Softmax (margin = 0.2)
* Large-Margin Finetune: random chunk(up to 6s), close speed perturb, margin (up to **0.5**), see config file for details.

| EER% | vox1-O | vox1-O-clean | vox1-E | vox1-E-clean | vox1-H | vox1-H-clean |
|:-----|:------:|:------------:|:------:|:------------:|:------:|:------------:|
| **Chunk-200**        |      |      |      |      |      |
|  Submean   | 0.954 | 0.819 | 1.129 | 0.998 | 2.025 | 1.9   |
|  Asnorm    | 0.822 | 0.691 | 1.079 | 0.948 | 1.906 | 1.773 |
| LM-Submean | 0.769 | 0.638 | 0.975 | 0.85  | 1.722 | 1.585 |
| LM-Asnorm  | 0.721 | 0.585 | 0.948 | 0.827 | 1.622 | 1.488 |
| **Chunk-300**        |      |      |      |      |      |
|  Submean   | 1.007 | 0.872 | 1.075 | 0.939 | 1.875 | 1.74  |
|  Asnorm    | 0.859 | 0.718 | 1.004 | 0.875 | 1.693 | 1.562 |
| LM-Submean | 0.891 | 0.739 | 1.008 | 0.876 | 1.739 | 1.607 |
| LM-Asnorm  | 0.822 | 0.67  | 0.985 | 0.849 | 1.657 | 1.529 |
<br/>

> **Note**: The paper adopts 3s as chunk length, and the results indicate that training with a longer chunk length pretrained model can yields better performance in hard trials *(Vox1-H)*. However, it may diminish the performace gain from Large-Margin Finetune. After Large-Margin Finetune, models pretrained with chunks of either 2s or 3s demonstrate similar performance levels.

### Results of ECAPA-TDNN (2023-12-27)
* Egs = Voxceleb2_dev(online random aug) + random chunk(2s)
* Optimization = [sgd (lr = 0.2 - 1e-6) + warm_cosine] x 4 GPUs (total batch-size=512) + AMP training 120 epochs + average best 5
* ECAPA-TDNN (channels = 1024) + AAM-Softmax (margin = 0.2)
* Large-Margin Finetune: random chunk(2s -> 6s), close speed perturb, margin (0.2 -> 0.5), see config file for details.

| EER% | vox1-O | vox1-O-clean | vox1-E | vox1-E-clean | vox1-H | vox1-H-clean |
|:-----|:------:|:------------:|:------:|:------------:|:------:|:------------:|
|  Submean   | 1.06  | 0.904 | 1.164 | 1.044 | 2.06  | 1.927 |
|  Asnorm    | 0.901 | 0.755 | 1.096 | 0.971 | 1.91  | 1.784 |
| LM-Submean | 0.795 | 0.66  | 0.992 | 0.859 | 1.743 | 1.606 |
| LM-Asnorm  | 0.758 | 0.617 | 0.95  | 0.818 | 1.644 | 1.505 |
<br/>

#### adamw VS sgd
Adamw converges quiet fast in the early stage, but its final performance is slightly worse than sgd, maybe it needs further training protocol tuning.
* Adamw: [(lr = 0.002 - 1e-8, weight decay = 0.05) + warm_cosine]
* Sgd: [(lr = 0.2 - 1e-6) + warm_cosine]
* AMP for 80 epochs + last checkpoint

| EER% | vox1-O | vox1-O-clean | vox1-E | vox1-E-clean | vox1-H | vox1-H-clean |
|:-----|:------:|:------------:|:------:|:------------:|:------:|:------------:|
|  Sgd-Submean   | 1.098 |  0.962 | 1.209 | 1.088 | 2.141 | 2.007 |
|  Sgd-Asnorm    | 0.938 |  0.819 | 1.14  | 1.014 | 1.986 | 1.851 |
|  Adamw-Submean | 1.071 |  0.941 | 1.248 | 1.129 | 2.355 | 2.219 |
|  Adamw-Asnorm  | 0.901 |  0.771 | 1.163 | 1.041 | 2.129 | 2.004 |
<br/>



### Results of ECAPA-TDNN (old version)
* Egs = Voxceleb2_dev(online random aug) + random chunk(2s)
* Optimization = [adamW (lr = 1e-8 - 1e-3) + cyclic for 3 cycle with triangular2 strategy] x 4 GPUs (total batch-size=512)
* ECAPA-TDNN (channels = 1024) + FC-BN + AAM-Softmax (margin = 0.2)
* Back-end = near + Cosine

| EER% | vox1-O | vox1-O-clean | vox1-E | vox1-E-clean | vox1-H | vox1-H-clean |
|:-----|:------:|:------------:|:------:|:------------:|:------:|:------------:|
|  Submean | 1.045 |  0.904 | 1.330 | 1.211 | 2.430 | 2.303 |
|  AS-Norm | 0.991 |  0.856 |   -   |   -   |   -   |   -   |
<br/>


### Results of Conformer (old version)
* Egs = Voxceleb2_dev(online random aug) + random chunk(3s)
* Optimization = [adamW (lr = 1e-6 - 1e-3) + 1cycle] x 4 GPUs (total batch-size=512)
* Conformer + FC-Swish-LN + ASP + FC-LN + AAM-Softmax (margin = 0.2)
* Back-end = near + Cosine
* LM: Large-Margin Fine-tune (margin: 0.2 --> 0.5, chunk: 6s)

| Config                       |        | vox1-O | vox1-O-clean | vox1-E | vox1-E-clean | vox1-H | vox1-H-clean |
|:---------------------------- |:------:|:------:|:------------:|:------:|:------------:|:------:|:------------:|
| 6L-256D-4H-4Sub (50 epochs)  |  Cosine  | 1.204 |  1.074 | 1.386 | 1.267 | 2.416 | 2.294 |
|                              |  AS-Norm | 1.092 |  0.952 |   -   |   -   |   -   |   -   |
| $\quad+$ SAM training        |  Cosine  | 1.103 |  0.984 | 1.350 | 1.234 | 2.380 | 2.257 |
|                              |  LM      | 1.034 |  0.899 | 1.181 | 1.060 | 2.079 | 1.953 |
|                              |  AS-Norm | 0.943 |  0.792 |   -   |   -   |   -   |   -   |
| 6L-256D-4H-2Sub (30 epochs)  |  Cosine  | 1.066 |  0.915 | 1.298 | 1.177 | 2.167 | 2.034 |
|                              |  LM      | 1.029 |  0.888 | 1.160 | 1.043 | 1.923 | 1.792 |
|                              |  AS-Norm | 0.949 |  0.792 |   -   |   -   |   -   |   -   |
<br/>

### Results of RTF (old version)
* RTF is evaluated on LibTorch-based runtime, see `subtools/runtime`
* One thread is used for CPU threading and TorchScript inference.
* CPU: Intel(R) Xeon(R) Gold 5218R CPU @ 2.10GHz.

| Model | Config | Params | RTF |
|:-----|:------  |:------:|:---:|
|  ResNet34  | base32 |  6.80M  | 0.090 |
|  ECAPA     | C1024  |  16.0M  | 0.071 |
|            | C512   |  6.53M  | 0.030 |
|  Conformer | 6L-256D-4H-4Sub  |  18.8M |   0.025   |
|            | 6L-256D-4H-2Sub  |  22.5M |   0.070   |
<br/>

### Effects of some tricks (2022-12-29)
* Egs = Voxceleb2_dev(online random aug) + random chunk(2s)
* Optimization = [adamW (lr = 1e-6 - 2e-3) + 1cycle] x 4 GPUs (total batch-size=512)
* ECAPA-TDNN (channels = 1024) + FC-BN + AM-Softmax (margin = 0.2), `subtools/pytorch/launcher/runEcapaXvector_roadmap.py`
* Back-end = near + Cosine

| Config | vox1-O | vox1-O-clean | vox1-E | vox1-E-clean | vox1-H | vox1-H-clean |
|:-----|:------:|:------------:|:------:|:------------:|:------:|:------------:|
|  Baseline | 1.188 |  1.048 | 1.313 | 1.190 | 2.369 | 2.241 |
|  + topk | 1.151 |  0.984 |   1.281   |   1.156   |   2.273   |   2.143   |
|  + subcenter | 1.113 |  0.984 |   1.255   |   1.139   |   2.239   |   2.108   |
|  + syncbn | 1.045 |  0.925 |   1.244   |   1.120   |   2.199   |   2.070   |
|  AM $\Rightarrow$ AAM | 1.103 |  0.952 |   1.213   |   1.084   |   2.133   |   1.994   |
|  + LM | 0.870 |  0.771 |   1.120   |   0.994   |   1.933   |   1.802   |
|  + mqmha(2q2h) | 0.832 |  0.713 |   1.103   |   0.978   |   1.925   |   1.796   |
|  + AS-Norm | 0.758 |  0.654 |   -   |   -   |   -   |   -   |
<br/>
