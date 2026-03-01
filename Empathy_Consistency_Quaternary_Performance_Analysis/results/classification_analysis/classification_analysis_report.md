# Classification Task Analysis Report

This report contains analysis results for four classification tasks:
1. Depression Binary (Yes/No)
2. Depression 4-Class (None/Mild/Moderate/Severe)
3. Suicide Risk Binary (Yes/No)
4. Suicide Risk 4-Class (No Risk/Low Risk/Medium Risk/High Risk)

---

## 1. Depression Binary Classification

### Haodf

#### BaichuanM2 (Without RAG)

**Confusion Matrix:**

| | Pred:No | Pred:Yes |
|---|---|---|
| True:No | 562 | 29 |
| True:Yes | 89 | 520 |

**Accuracy:** 0.9017

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No | 0.8633 | 0.9509 | 0.9050 | 591 |
| Yes | 0.9472 | 0.8539 | 0.8981 | 609 |
| Macro Avg | 0.9052 | 0.9024 | 0.9015 | - |
| Weighted Avg | 0.9059 | 0.9017 | 0.9015 | - |

#### BaichuanM2 (With RAG)

**Confusion Matrix:**

| | Pred:No | Pred:Yes |
|---|---|---|
| True:No | 625 | 123 |
| True:Yes | 26 | 426 |

**Accuracy:** 0.8758

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No | 0.9601 | 0.8356 | 0.8935 | 748 |
| Yes | 0.7760 | 0.9425 | 0.8511 | 452 |
| Macro Avg | 0.8680 | 0.8890 | 0.8723 | - |
| Weighted Avg | 0.8907 | 0.8758 | 0.8775 | - |

#### DeepSeekR1 (Without RAG)

**Confusion Matrix:**

| | Pred:No | Pred:Yes |
|---|---|---|
| True:No | 534 | 5 |
| True:Yes | 117 | 544 |

**Accuracy:** 0.8983

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No | 0.8203 | 0.9907 | 0.8975 | 539 |
| Yes | 0.9909 | 0.8230 | 0.8992 | 661 |
| Macro Avg | 0.9056 | 0.9069 | 0.8983 | - |
| Weighted Avg | 0.9143 | 0.8983 | 0.8984 | - |

#### DeepSeekR1 (With RAG)

**Confusion Matrix:**

| | Pred:No | Pred:Yes |
|---|---|---|
| True:No | 551 | 30 |
| True:Yes | 100 | 519 |

**Accuracy:** 0.8917

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No | 0.8464 | 0.9484 | 0.8945 | 581 |
| Yes | 0.9454 | 0.8384 | 0.8887 | 619 |
| Macro Avg | 0.8959 | 0.8934 | 0.8916 | - |
| Weighted Avg | 0.8974 | 0.8917 | 0.8915 | - |

#### Qwen3235BA22BInstruct2507 (Without RAG)

**Confusion Matrix:**

| | Pred:No | Pred:Yes |
|---|---|---|
| True:No | 540 | 12 |
| True:Yes | 111 | 537 |

**Accuracy:** 0.8975

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No | 0.8295 | 0.9783 | 0.8978 | 552 |
| Yes | 0.9781 | 0.8287 | 0.8972 | 648 |
| Macro Avg | 0.9038 | 0.9035 | 0.8975 | - |
| Weighted Avg | 0.9098 | 0.8975 | 0.8975 | - |

#### Qwen3235BA22BInstruct2507 (With RAG)

**Confusion Matrix:**

| | Pred:No | Pred:Yes |
|---|---|---|
| True:No | 603 | 59 |
| True:Yes | 48 | 490 |

**Accuracy:** 0.9108

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No | 0.9263 | 0.9109 | 0.9185 | 662 |
| Yes | 0.8925 | 0.9108 | 0.9016 | 538 |
| Macro Avg | 0.9094 | 0.9108 | 0.9100 | - |
| Weighted Avg | 0.9111 | 0.9108 | 0.9109 | - |

### Standard

#### BaichuanM2 (Without RAG)

**Confusion Matrix:**

| | Pred:No | Pred:Yes |
|---|---|---|
| True:No | 263 | 20 |
| True:Yes | 19 | 160 |

**Accuracy:** 0.9156

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No | 0.9326 | 0.9293 | 0.9310 | 283 |
| Yes | 0.8889 | 0.8939 | 0.8914 | 179 |
| Macro Avg | 0.9108 | 0.9116 | 0.9112 | - |
| Weighted Avg | 0.9157 | 0.9156 | 0.9156 | - |

#### BaichuanM2 (With RAG)

**Confusion Matrix:**

| | Pred:No | Pred:Yes |
|---|---|---|
| True:No | 262 | 27 |
| True:Yes | 20 | 153 |

**Accuracy:** 0.8983

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No | 0.9291 | 0.9066 | 0.9177 | 289 |
| Yes | 0.8500 | 0.8844 | 0.8669 | 173 |
| Macro Avg | 0.8895 | 0.8955 | 0.8923 | - |
| Weighted Avg | 0.8995 | 0.8983 | 0.8987 | - |

#### DeepSeekR1 (Without RAG)

**Confusion Matrix:**

| | Pred:No | Pred:Yes |
|---|---|---|
| True:No | 243 | 11 |
| True:Yes | 39 | 169 |

**Accuracy:** 0.8918

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No | 0.8617 | 0.9567 | 0.9067 | 254 |
| Yes | 0.9389 | 0.8125 | 0.8711 | 208 |
| Macro Avg | 0.9003 | 0.8846 | 0.8889 | - |
| Weighted Avg | 0.8965 | 0.8918 | 0.8907 | - |

#### DeepSeekR1 (With RAG)

**Confusion Matrix:**

| | Pred:No | Pred:Yes |
|---|---|---|
| True:No | 245 | 11 |
| True:Yes | 37 | 169 |

**Accuracy:** 0.8961

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No | 0.8688 | 0.9570 | 0.9108 | 256 |
| Yes | 0.9389 | 0.8204 | 0.8756 | 206 |
| Macro Avg | 0.9038 | 0.8887 | 0.8932 | - |
| Weighted Avg | 0.9000 | 0.8961 | 0.8951 | - |

#### Qwen3235BA22BInstruct2507 (Without RAG)

**Confusion Matrix:**

| | Pred:No | Pred:Yes |
|---|---|---|
| True:No | 244 | 11 |
| True:Yes | 38 | 169 |

**Accuracy:** 0.8939

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No | 0.8652 | 0.9569 | 0.9088 | 255 |
| Yes | 0.9389 | 0.8164 | 0.8734 | 207 |
| Macro Avg | 0.9021 | 0.8866 | 0.8911 | - |
| Weighted Avg | 0.8982 | 0.8939 | 0.8929 | - |

#### Qwen3235BA22BInstruct2507 (With RAG)

**Confusion Matrix:**

| | Pred:No | Pred:Yes |
|---|---|---|
| True:No | 277 | 20 |
| True:Yes | 5 | 160 |

**Accuracy:** 0.9459

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No | 0.9823 | 0.9327 | 0.9568 | 297 |
| Yes | 0.8889 | 0.9697 | 0.9275 | 165 |
| Macro Avg | 0.9356 | 0.9512 | 0.9422 | - |
| Weighted Avg | 0.9489 | 0.9459 | 0.9464 | - |

## 2. Depression 4-Class Classification

### Haodf

#### BaichuanM2 (Without RAG)

**Confusion Matrix:**

| | Pred:None | Pred:Mild | Pred:Moderate | Pred:Severe |
|---|---|---|---|---|
| True:None | 477 | 98 | 50 | 26 |
| True:Mild | 0 | 66 | 9 | 0 |
| True:Moderate | 0 | 12 | 165 | 0 |
| True:Severe | 0 | 0 | 111 | 186 |

**Accuracy:** 0.7450

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| None | 1.0000 | 0.7327 | 0.8457 | 651 |
| Mild | 0.3750 | 0.8800 | 0.5259 | 75 |
| Moderate | 0.4925 | 0.9322 | 0.6445 | 177 |
| Severe | 0.8774 | 0.6263 | 0.7308 | 297 |
| Macro Avg | 0.6862 | 0.7928 | 0.6868 | - |
| Weighted Avg | 0.8557 | 0.7450 | 0.7676 | - |

#### BaichuanM2 (With RAG)

**Confusion Matrix:**

| | Pred:None | Pred:Mild | Pred:Moderate | Pred:Severe |
|---|---|---|---|---|
| True:None | 616 | 6 | 21 | 8 |
| True:Mild | 54 | 19 | 2 | 0 |
| True:Moderate | 13 | 14 | 145 | 5 |
| True:Severe | 4 | 2 | 92 | 199 |

**Accuracy:** 0.8158

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| None | 0.8967 | 0.9462 | 0.9208 | 651 |
| Mild | 0.4634 | 0.2533 | 0.3276 | 75 |
| Moderate | 0.5577 | 0.8192 | 0.6636 | 177 |
| Severe | 0.9387 | 0.6700 | 0.7819 | 297 |
| Macro Avg | 0.7141 | 0.6722 | 0.6735 | - |
| Weighted Avg | 0.8300 | 0.8158 | 0.8114 | - |

#### DeepSeekR1 (Without RAG)

**Confusion Matrix:**

| | Pred:None | Pred:Mild | Pred:Moderate | Pred:Severe |
|---|---|---|---|---|
| True:None | 448 | 87 | 73 | 43 |
| True:Mild | 0 | 49 | 26 | 0 |
| True:Moderate | 0 | 0 | 170 | 7 |
| True:Severe | 0 | 0 | 70 | 227 |

**Accuracy:** 0.7450

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| None | 1.0000 | 0.6882 | 0.8153 | 651 |
| Mild | 0.3603 | 0.6533 | 0.4645 | 75 |
| Moderate | 0.5015 | 0.9605 | 0.6589 | 177 |
| Severe | 0.8195 | 0.7643 | 0.7909 | 297 |
| Macro Avg | 0.6703 | 0.7666 | 0.6824 | - |
| Weighted Avg | 0.8418 | 0.7450 | 0.7643 | - |

#### DeepSeekR1 (With RAG)

**Confusion Matrix:**

| | Pred:None | Pred:Mild | Pred:Moderate | Pred:Severe |
|---|---|---|---|---|
| True:None | 525 | 36 | 51 | 39 |
| True:Mild | 17 | 43 | 15 | 0 |
| True:Moderate | 1 | 7 | 164 | 5 |
| True:Severe | 0 | 1 | 60 | 236 |

**Accuracy:** 0.8067

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| None | 0.9669 | 0.8065 | 0.8794 | 651 |
| Mild | 0.4943 | 0.5733 | 0.5309 | 75 |
| Moderate | 0.5655 | 0.9266 | 0.7024 | 177 |
| Severe | 0.8429 | 0.7946 | 0.8180 | 297 |
| Macro Avg | 0.7174 | 0.7752 | 0.7327 | - |
| Weighted Avg | 0.8474 | 0.8067 | 0.8163 | - |

#### Qwen3235BA22BInstruct2507 (Without RAG)

**Confusion Matrix:**

| | Pred:None | Pred:Mild | Pred:Moderate | Pred:Severe |
|---|---|---|---|---|
| True:None | 377 | 158 | 83 | 33 |
| True:Mild | 0 | 66 | 9 | 0 |
| True:Moderate | 0 | 5 | 172 | 0 |
| True:Severe | 0 | 0 | 79 | 218 |

**Accuracy:** 0.6942

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| None | 1.0000 | 0.5791 | 0.7335 | 651 |
| Mild | 0.2882 | 0.8800 | 0.4342 | 75 |
| Moderate | 0.5015 | 0.9718 | 0.6615 | 177 |
| Severe | 0.8685 | 0.7340 | 0.7956 | 297 |
| Macro Avg | 0.6645 | 0.7912 | 0.6562 | - |
| Weighted Avg | 0.8494 | 0.6942 | 0.7195 | - |

#### Qwen3235BA22BInstruct2507 (With RAG)

**Confusion Matrix:**

| | Pred:None | Pred:Mild | Pred:Moderate | Pred:Severe |
|---|---|---|---|---|
| True:None | 579 | 19 | 19 | 34 |
| True:Mild | 4 | 68 | 3 | 0 |
| True:Moderate | 1 | 11 | 151 | 14 |
| True:Severe | 0 | 0 | 24 | 273 |

**Accuracy:** 0.8925

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| None | 0.9914 | 0.8894 | 0.9377 | 651 |
| Mild | 0.6939 | 0.9067 | 0.7861 | 75 |
| Moderate | 0.7665 | 0.8531 | 0.8075 | 177 |
| Severe | 0.8505 | 0.9192 | 0.8835 | 297 |
| Macro Avg | 0.8256 | 0.8921 | 0.8537 | - |
| Weighted Avg | 0.9048 | 0.8925 | 0.8956 | - |

### Standard

#### BaichuanM2 (Without RAG)

**Confusion Matrix:**

| | Pred:None | Pred:Mild | Pred:Moderate | Pred:Severe |
|---|---|---|---|---|
| True:None | 211 | 49 | 22 | 0 |
| True:Mild | 0 | 23 | 1 | 0 |
| True:Moderate | 0 | 15 | 56 | 1 |
| True:Severe | 0 | 0 | 74 | 10 |

**Accuracy:** 0.6494

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| None | 1.0000 | 0.7482 | 0.8560 | 282 |
| Mild | 0.2644 | 0.9583 | 0.4144 | 24 |
| Moderate | 0.3660 | 0.7778 | 0.4978 | 72 |
| Severe | 0.9091 | 0.1190 | 0.2105 | 84 |
| Macro Avg | 0.6349 | 0.6508 | 0.4947 | - |
| Weighted Avg | 0.8465 | 0.6494 | 0.6599 | - |

#### BaichuanM2 (With RAG)

**Confusion Matrix:**

| | Pred:None | Pred:Mild | Pred:Moderate | Pred:Severe |
|---|---|---|---|---|
| True:None | 252 | 9 | 19 | 2 |
| True:Mild | 13 | 2 | 9 | 0 |
| True:Moderate | 7 | 3 | 53 | 9 |
| True:Severe | 0 | 0 | 27 | 57 |

**Accuracy:** 0.7879

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| None | 0.9265 | 0.8936 | 0.9097 | 282 |
| Mild | 0.1429 | 0.0833 | 0.1053 | 24 |
| Moderate | 0.4907 | 0.7361 | 0.5889 | 72 |
| Severe | 0.8382 | 0.6786 | 0.7500 | 84 |
| Macro Avg | 0.5996 | 0.5979 | 0.5885 | - |
| Weighted Avg | 0.8018 | 0.7879 | 0.7889 | - |

#### DeepSeekR1 (Without RAG)

**Confusion Matrix:**

| | Pred:None | Pred:Mild | Pred:Moderate | Pred:Severe |
|---|---|---|---|---|
| True:None | 224 | 28 | 30 | 0 |
| True:Mild | 0 | 16 | 8 | 0 |
| True:Moderate | 0 | 0 | 72 | 0 |
| True:Severe | 0 | 0 | 72 | 12 |

**Accuracy:** 0.7013

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| None | 1.0000 | 0.7943 | 0.8854 | 282 |
| Mild | 0.3636 | 0.6667 | 0.4706 | 24 |
| Moderate | 0.3956 | 1.0000 | 0.5669 | 72 |
| Severe | 1.0000 | 0.1429 | 0.2500 | 84 |
| Macro Avg | 0.6898 | 0.6510 | 0.5432 | - |
| Weighted Avg | 0.8728 | 0.7013 | 0.6987 | - |

#### DeepSeekR1 (With RAG)

**Confusion Matrix:**

| | Pred:None | Pred:Mild | Pred:Moderate | Pred:Severe |
|---|---|---|---|---|
| True:None | 238 | 11 | 32 | 1 |
| True:Mild | 2 | 11 | 11 | 0 |
| True:Moderate | 2 | 0 | 68 | 2 |
| True:Severe | 0 | 0 | 64 | 20 |

**Accuracy:** 0.7294

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| None | 0.9835 | 0.8440 | 0.9084 | 282 |
| Mild | 0.5000 | 0.4583 | 0.4783 | 24 |
| Moderate | 0.3886 | 0.9444 | 0.5506 | 72 |
| Severe | 0.8696 | 0.2381 | 0.3738 | 84 |
| Macro Avg | 0.6854 | 0.6212 | 0.5778 | - |
| Weighted Avg | 0.8449 | 0.7294 | 0.7331 | - |

#### Qwen3235BA22BInstruct2507 (Without RAG)

**Confusion Matrix:**

| | Pred:None | Pred:Mild | Pred:Moderate | Pred:Severe |
|---|---|---|---|---|
| True:None | 185 | 51 | 46 | 0 |
| True:Mild | 0 | 16 | 8 | 0 |
| True:Moderate | 0 | 4 | 68 | 0 |
| True:Severe | 0 | 0 | 72 | 12 |

**Accuracy:** 0.6082

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| None | 1.0000 | 0.6560 | 0.7923 | 282 |
| Mild | 0.2254 | 0.6667 | 0.3368 | 24 |
| Moderate | 0.3505 | 0.9444 | 0.5113 | 72 |
| Severe | 1.0000 | 0.1429 | 0.2500 | 84 |
| Macro Avg | 0.6440 | 0.6025 | 0.4726 | - |
| Weighted Avg | 0.8585 | 0.6082 | 0.6262 | - |

#### Qwen3235BA22BInstruct2507 (With RAG)

**Confusion Matrix:**

| | Pred:None | Pred:Mild | Pred:Moderate | Pred:Severe |
|---|---|---|---|---|
| True:None | 257 | 10 | 15 | 0 |
| True:Mild | 2 | 19 | 3 | 0 |
| True:Moderate | 0 | 3 | 69 | 0 |
| True:Severe | 0 | 0 | 34 | 50 |

**Accuracy:** 0.8550

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| None | 0.9923 | 0.9113 | 0.9501 | 282 |
| Mild | 0.5938 | 0.7917 | 0.6786 | 24 |
| Moderate | 0.5702 | 0.9583 | 0.7150 | 72 |
| Severe | 1.0000 | 0.5952 | 0.7463 | 84 |
| Macro Avg | 0.7891 | 0.8141 | 0.7725 | - |
| Weighted Avg | 0.9072 | 0.8550 | 0.8623 | - |

## 3. Suicide Risk Binary Classification

### Haodf

#### BaichuanM2 (Without RAG)

**Confusion Matrix:**

| | Pred:No Risk | Pred:Has Risk |
|---|---|---|
| True:No Risk | 701 | 85 |
| True:Has Risk | 40 | 374 |

**Accuracy:** 0.8958

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Risk | 0.9460 | 0.8919 | 0.9181 | 786 |
| Has Risk | 0.8148 | 0.9034 | 0.8568 | 414 |
| Macro Avg | 0.8804 | 0.8976 | 0.8875 | - |
| Weighted Avg | 0.9008 | 0.8958 | 0.8970 | - |

#### BaichuanM2 (With RAG)

**Confusion Matrix:**

| | Pred:No Risk | Pred:Has Risk |
|---|---|---|
| True:No Risk | 715 | 73 |
| True:Has Risk | 26 | 386 |

**Accuracy:** 0.9175

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Risk | 0.9649 | 0.9074 | 0.9353 | 788 |
| Has Risk | 0.8410 | 0.9369 | 0.8863 | 412 |
| Macro Avg | 0.9029 | 0.9221 | 0.9108 | - |
| Weighted Avg | 0.9224 | 0.9175 | 0.9185 | - |

#### DeepSeekR1 (Without RAG)

**Confusion Matrix:**

| | Pred:No Risk | Pred:Has Risk |
|---|---|---|
| True:No Risk | 700 | 87 |
| True:Has Risk | 41 | 372 |

**Accuracy:** 0.8933

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Risk | 0.9447 | 0.8895 | 0.9162 | 787 |
| Has Risk | 0.8105 | 0.9007 | 0.8532 | 413 |
| Macro Avg | 0.8776 | 0.8951 | 0.8847 | - |
| Weighted Avg | 0.8985 | 0.8933 | 0.8945 | - |

#### DeepSeekR1 (With RAG)

**Confusion Matrix:**

| | Pred:No Risk | Pred:Has Risk |
|---|---|---|
| True:No Risk | 704 | 78 |
| True:Has Risk | 37 | 381 |

**Accuracy:** 0.9042

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Risk | 0.9501 | 0.9003 | 0.9245 | 782 |
| Has Risk | 0.8301 | 0.9115 | 0.8689 | 418 |
| Macro Avg | 0.8901 | 0.9059 | 0.8967 | - |
| Weighted Avg | 0.9083 | 0.9042 | 0.9051 | - |

#### Qwen3235BA22BInstruct2507 (Without RAG)

**Confusion Matrix:**

| | Pred:No Risk | Pred:Has Risk |
|---|---|---|
| True:No Risk | 656 | 77 |
| True:Has Risk | 85 | 382 |

**Accuracy:** 0.8650

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Risk | 0.8853 | 0.8950 | 0.8901 | 733 |
| Has Risk | 0.8322 | 0.8180 | 0.8251 | 467 |
| Macro Avg | 0.8588 | 0.8565 | 0.8576 | - |
| Weighted Avg | 0.8646 | 0.8650 | 0.8648 | - |

#### Qwen3235BA22BInstruct2507 (With RAG)

**Confusion Matrix:**

| | Pred:No Risk | Pred:Has Risk |
|---|---|---|
| True:No Risk | 697 | 94 |
| True:Has Risk | 44 | 365 |

**Accuracy:** 0.8850

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Risk | 0.9406 | 0.8812 | 0.9099 | 791 |
| Has Risk | 0.7952 | 0.8924 | 0.8410 | 409 |
| Macro Avg | 0.8679 | 0.8868 | 0.8755 | - |
| Weighted Avg | 0.8911 | 0.8850 | 0.8864 | - |

### Standard

#### BaichuanM2 (Without RAG)

**Confusion Matrix:**

| | Pred:No Risk | Pred:Has Risk |
|---|---|---|
| True:No Risk | 306 | 93 |
| True:Has Risk | 3 | 60 |

**Accuracy:** 0.7922

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Risk | 0.9903 | 0.7669 | 0.8644 | 399 |
| Has Risk | 0.3922 | 0.9524 | 0.5556 | 63 |
| Macro Avg | 0.6912 | 0.8596 | 0.7100 | - |
| Weighted Avg | 0.9087 | 0.7922 | 0.8223 | - |

#### BaichuanM2 (With RAG)

**Confusion Matrix:**

| | Pred:No Risk | Pred:Has Risk |
|---|---|---|
| True:No Risk | 298 | 27 |
| True:Has Risk | 11 | 126 |

**Accuracy:** 0.9177

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Risk | 0.9644 | 0.9169 | 0.9401 | 325 |
| Has Risk | 0.8235 | 0.9197 | 0.8690 | 137 |
| Macro Avg | 0.8940 | 0.9183 | 0.9045 | - |
| Weighted Avg | 0.9226 | 0.9177 | 0.9190 | - |

#### DeepSeekR1 (Without RAG)

**Confusion Matrix:**

| | Pred:No Risk | Pred:Has Risk |
|---|---|---|
| True:No Risk | 308 | 97 |
| True:Has Risk | 1 | 56 |

**Accuracy:** 0.7879

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Risk | 0.9968 | 0.7605 | 0.8627 | 405 |
| Has Risk | 0.3660 | 0.9825 | 0.5333 | 57 |
| Macro Avg | 0.6814 | 0.8715 | 0.6980 | - |
| Weighted Avg | 0.9189 | 0.7879 | 0.8221 | - |

#### DeepSeekR1 (With RAG)

**Confusion Matrix:**

| | Pred:No Risk | Pred:Has Risk |
|---|---|---|
| True:No Risk | 304 | 36 |
| True:Has Risk | 5 | 117 |

**Accuracy:** 0.9113

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Risk | 0.9838 | 0.8941 | 0.9368 | 340 |
| Has Risk | 0.7647 | 0.9590 | 0.8509 | 122 |
| Macro Avg | 0.8743 | 0.9266 | 0.8939 | - |
| Weighted Avg | 0.9260 | 0.9113 | 0.9141 | - |

#### Qwen3235BA22BInstruct2507 (Without RAG)

**Confusion Matrix:**

| | Pred:No Risk | Pred:Has Risk |
|---|---|---|
| True:No Risk | 306 | 80 |
| True:Has Risk | 3 | 73 |

**Accuracy:** 0.8203

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Risk | 0.9903 | 0.7927 | 0.8806 | 386 |
| Has Risk | 0.4771 | 0.9605 | 0.6376 | 76 |
| Macro Avg | 0.7337 | 0.8766 | 0.7591 | - |
| Weighted Avg | 0.9059 | 0.8203 | 0.8406 | - |

#### Qwen3235BA22BInstruct2507 (With RAG)

**Confusion Matrix:**

| | Pred:No Risk | Pred:Has Risk |
|---|---|---|
| True:No Risk | 309 | 100 |
| True:Has Risk | 0 | 53 |

**Accuracy:** 0.7835

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Risk | 1.0000 | 0.7555 | 0.8607 | 409 |
| Has Risk | 0.3464 | 1.0000 | 0.5146 | 53 |
| Macro Avg | 0.6732 | 0.8778 | 0.6876 | - |
| Weighted Avg | 0.9250 | 0.7835 | 0.8210 | - |

## 4. Suicide Risk 4-Class Classification

### Haodf

#### BaichuanM2 (Without RAG)

**Confusion Matrix:**

| | Pred:No Risk | Pred:Low Risk | Pred:Medium Risk | Pred:High Risk |
|---|---|---|---|---|
| True:No Risk | 672 | 43 | 16 | 10 |
| True:Low Risk | 43 | 119 | 18 | 0 |
| True:Medium Risk | 2 | 38 | 72 | 44 |
| True:High Risk | 0 | 0 | 6 | 117 |

**Accuracy:** 0.8167

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Risk | 0.9372 | 0.9069 | 0.9218 | 741 |
| Low Risk | 0.5950 | 0.6611 | 0.6263 | 180 |
| Medium Risk | 0.6429 | 0.4615 | 0.5373 | 156 |
| High Risk | 0.6842 | 0.9512 | 0.7959 | 123 |
| Macro Avg | 0.7148 | 0.7452 | 0.7203 | - |
| Weighted Avg | 0.8217 | 0.8167 | 0.8146 | - |

#### BaichuanM2 (With RAG)

**Confusion Matrix:**

| | Pred:No Risk | Pred:Low Risk | Pred:Medium Risk | Pred:High Risk |
|---|---|---|---|---|
| True:No Risk | 708 | 27 | 4 | 2 |
| True:Low Risk | 65 | 115 | 0 | 0 |
| True:Medium Risk | 6 | 66 | 75 | 9 |
| True:High Risk | 0 | 0 | 62 | 61 |

**Accuracy:** 0.7992

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Risk | 0.9089 | 0.9555 | 0.9316 | 741 |
| Low Risk | 0.5529 | 0.6389 | 0.5928 | 180 |
| Medium Risk | 0.5319 | 0.4808 | 0.5051 | 156 |
| High Risk | 0.8472 | 0.4959 | 0.6256 | 123 |
| Macro Avg | 0.7102 | 0.6428 | 0.6638 | - |
| Weighted Avg | 0.8001 | 0.7992 | 0.7940 | - |

#### DeepSeekR1 (Without RAG)

**Confusion Matrix:**

| | Pred:No Risk | Pred:Low Risk | Pred:Medium Risk | Pred:High Risk |
|---|---|---|---|---|
| True:No Risk | 651 | 43 | 40 | 7 |
| True:Low Risk | 32 | 117 | 31 | 0 |
| True:Medium Risk | 0 | 13 | 90 | 53 |
| True:High Risk | 0 | 0 | 3 | 120 |

**Accuracy:** 0.8150

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Risk | 0.9531 | 0.8785 | 0.9143 | 741 |
| Low Risk | 0.6763 | 0.6500 | 0.6629 | 180 |
| Medium Risk | 0.5488 | 0.5769 | 0.5625 | 156 |
| High Risk | 0.6667 | 0.9756 | 0.7921 | 123 |
| Macro Avg | 0.7112 | 0.7703 | 0.7329 | - |
| Weighted Avg | 0.8297 | 0.8150 | 0.8183 | - |

#### DeepSeekR1 (With RAG)

**Confusion Matrix:**

| | Pred:No Risk | Pred:Low Risk | Pred:Medium Risk | Pred:High Risk |
|---|---|---|---|---|
| True:No Risk | 701 | 33 | 5 | 2 |
| True:Low Risk | 56 | 124 | 0 | 0 |
| True:Medium Risk | 3 | 61 | 86 | 6 |
| True:High Risk | 0 | 0 | 49 | 74 |

**Accuracy:** 0.8208

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Risk | 0.9224 | 0.9460 | 0.9340 | 741 |
| Low Risk | 0.5688 | 0.6889 | 0.6231 | 180 |
| Medium Risk | 0.6143 | 0.5513 | 0.5811 | 156 |
| High Risk | 0.9024 | 0.6016 | 0.7220 | 123 |
| Macro Avg | 0.7520 | 0.6970 | 0.7150 | - |
| Weighted Avg | 0.8272 | 0.8208 | 0.8198 | - |

#### Qwen3235BA22BInstruct2507 (Without RAG)

**Confusion Matrix:**

| | Pred:No Risk | Pred:Low Risk | Pred:Medium Risk | Pred:High Risk |
|---|---|---|---|---|
| True:No Risk | 588 | 52 | 76 | 25 |
| True:Low Risk | 3 | 112 | 65 | 0 |
| True:Medium Risk | 0 | 3 | 90 | 63 |
| True:High Risk | 0 | 0 | 2 | 121 |

**Accuracy:** 0.7592

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Risk | 0.9949 | 0.7935 | 0.8829 | 741 |
| Low Risk | 0.6707 | 0.6222 | 0.6455 | 180 |
| Medium Risk | 0.3863 | 0.5769 | 0.4627 | 156 |
| High Risk | 0.5789 | 0.9837 | 0.7289 | 123 |
| Macro Avg | 0.6577 | 0.7441 | 0.6800 | - |
| Weighted Avg | 0.8245 | 0.7592 | 0.7769 | - |

#### Qwen3235BA22BInstruct2507 (With RAG)

**Confusion Matrix:**

| | Pred:No Risk | Pred:Low Risk | Pred:Medium Risk | Pred:High Risk |
|---|---|---|---|---|
| True:No Risk | 674 | 32 | 28 | 7 |
| True:Low Risk | 18 | 155 | 7 | 0 |
| True:Medium Risk | 0 | 22 | 128 | 6 |
| True:High Risk | 0 | 0 | 26 | 97 |

**Accuracy:** 0.8783

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Risk | 0.9740 | 0.9096 | 0.9407 | 741 |
| Low Risk | 0.7416 | 0.8611 | 0.7969 | 180 |
| Medium Risk | 0.6772 | 0.8205 | 0.7420 | 156 |
| High Risk | 0.8818 | 0.7886 | 0.8326 | 123 |
| Macro Avg | 0.8187 | 0.8450 | 0.8281 | - |
| Weighted Avg | 0.8911 | 0.8783 | 0.8822 | - |

### Standard

#### BaichuanM2 (Without RAG)

**Confusion Matrix:**

| | Pred:No Risk | Pred:Low Risk | Pred:Medium Risk | Pred:High Risk |
|---|---|---|---|---|
| True:No Risk | 296 | 11 | 2 | 0 |
| True:Low Risk | 31 | 89 | 6 | 0 |
| True:Medium Risk | 0 | 10 | 16 | 1 |
| True:High Risk | 0 | 0 | 0 | 0 |

**Accuracy:** 0.8680

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Risk | 0.9052 | 0.9579 | 0.9308 | 309 |
| Low Risk | 0.8091 | 0.7063 | 0.7542 | 126 |
| Medium Risk | 0.6667 | 0.5926 | 0.6275 | 27 |
| High Risk | 0.0000 | 0.0000 | 0.0000 | 0 |
| Macro Avg | 0.5952 | 0.5642 | 0.5781 | - |
| Weighted Avg | 0.8650 | 0.8680 | 0.8649 | - |

#### BaichuanM2 (With RAG)

**Confusion Matrix:**

| | Pred:No Risk | Pred:Low Risk | Pred:Medium Risk | Pred:High Risk |
|---|---|---|---|---|
| True:No Risk | 300 | 9 | 0 | 0 |
| True:Low Risk | 19 | 107 | 0 | 0 |
| True:Medium Risk | 0 | 12 | 15 | 0 |
| True:High Risk | 0 | 0 | 0 | 0 |

**Accuracy:** 0.9134

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Risk | 0.9404 | 0.9709 | 0.9554 | 309 |
| Low Risk | 0.8359 | 0.8492 | 0.8425 | 126 |
| Medium Risk | 1.0000 | 0.5556 | 0.7143 | 27 |
| High Risk | 0.0000 | 0.0000 | 0.0000 | 0 |
| Macro Avg | 0.9255 | 0.7919 | 0.8374 | - |
| Weighted Avg | 0.9154 | 0.9134 | 0.9105 | - |

#### DeepSeekR1 (Without RAG)

**Confusion Matrix:**

| | Pred:No Risk | Pred:Low Risk | Pred:Medium Risk | Pred:High Risk |
|---|---|---|---|---|
| True:No Risk | 301 | 8 | 0 | 0 |
| True:Low Risk | 20 | 100 | 6 | 0 |
| True:Medium Risk | 0 | 6 | 14 | 7 |
| True:High Risk | 0 | 0 | 0 | 0 |

**Accuracy:** 0.8983

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Risk | 0.9377 | 0.9741 | 0.9556 | 309 |
| Low Risk | 0.8772 | 0.7937 | 0.8333 | 126 |
| Medium Risk | 0.7000 | 0.5185 | 0.5957 | 27 |
| High Risk | 0.0000 | 0.0000 | 0.0000 | 0 |
| Macro Avg | 0.6287 | 0.5716 | 0.5962 | - |
| Weighted Avg | 0.9073 | 0.8983 | 0.9012 | - |

#### DeepSeekR1 (With RAG)

**Confusion Matrix:**

| | Pred:No Risk | Pred:Low Risk | Pred:Medium Risk | Pred:High Risk |
|---|---|---|---|---|
| True:No Risk | 299 | 10 | 0 | 0 |
| True:Low Risk | 10 | 116 | 0 | 0 |
| True:Medium Risk | 0 | 13 | 14 | 0 |
| True:High Risk | 0 | 0 | 0 | 0 |

**Accuracy:** 0.9286

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Risk | 0.9676 | 0.9676 | 0.9676 | 309 |
| Low Risk | 0.8345 | 0.9206 | 0.8755 | 126 |
| Medium Risk | 1.0000 | 0.5185 | 0.6829 | 27 |
| High Risk | 0.0000 | 0.0000 | 0.0000 | 0 |
| Macro Avg | 0.9341 | 0.8023 | 0.8420 | - |
| Weighted Avg | 0.9332 | 0.9286 | 0.9259 | - |

#### Qwen3235BA22BInstruct2507 (Without RAG)

**Confusion Matrix:**

| | Pred:No Risk | Pred:Low Risk | Pred:Medium Risk | Pred:High Risk |
|---|---|---|---|---|
| True:No Risk | 272 | 31 | 6 | 0 |
| True:Low Risk | 4 | 92 | 30 | 0 |
| True:Medium Risk | 0 | 0 | 18 | 9 |
| True:High Risk | 0 | 0 | 0 | 0 |

**Accuracy:** 0.8268

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Risk | 0.9855 | 0.8803 | 0.9299 | 309 |
| Low Risk | 0.7480 | 0.7302 | 0.7390 | 126 |
| Medium Risk | 0.3333 | 0.6667 | 0.4444 | 27 |
| High Risk | 0.0000 | 0.0000 | 0.0000 | 0 |
| Macro Avg | 0.5167 | 0.5693 | 0.5283 | - |
| Weighted Avg | 0.8826 | 0.8268 | 0.8495 | - |

#### Qwen3235BA22BInstruct2507 (With RAG)

**Confusion Matrix:**

| | Pred:No Risk | Pred:Low Risk | Pred:Medium Risk | Pred:High Risk |
|---|---|---|---|---|
| True:No Risk | 297 | 12 | 0 | 0 |
| True:Low Risk | 3 | 123 | 0 | 0 |
| True:Medium Risk | 0 | 6 | 21 | 0 |
| True:High Risk | 0 | 0 | 0 | 0 |

**Accuracy:** 0.9545

**Per-Class Metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No Risk | 0.9900 | 0.9612 | 0.9754 | 309 |
| Low Risk | 0.8723 | 0.9762 | 0.9213 | 126 |
| Medium Risk | 1.0000 | 0.7778 | 0.8750 | 27 |
| High Risk | 0.0000 | 0.0000 | 0.0000 | 0 |
| Macro Avg | 0.9541 | 0.9050 | 0.9239 | - |
| Weighted Avg | 0.9585 | 0.9545 | 0.9548 | - |

