# D-DARTS: Nested/Distributed Differentiable Architecture Search 

*: This is the official implementation of D-DARTS.

*Differentiable ARchiTecture Search (DARTS) is one of the most trending Neural Architecture Search (NAS) methods, drastically reducing search cost by resorting to Stochastic Gradient Descent (SGD) and weight-sharing. However, it also dramatically reduces the search space, thus excluding potential promising architectures from being discovered. In this article, we propose D-DARTS, a novel solution that addresses this problem by nesting several neural networks at the cell level instead of using weight-sharing to produce more diversified and specialized architectures. Moreover, we introduce a novel algorithm that can derive deeper architectures from a few trained cells, increasing performance and saving computation time. In addition, we also present an alternative search space (denoted DARTOpti) in which we optimize existing handcrafted architectures such as ResNet rather than starting from a blank canvas. This approach is accompanied by a novel metric that measures the distance between architectures inside our custom search space. Our solution achieves state-of-the-art on CIFAR-10, CIFAR-100, and ImageNet while featuring a search cost significantly lower than previous differentiable NAS approaches.*

## User Guide 

### Prerequisites

> Python >= 3.6

`pip install -r requirements.txt`
OR
`conda install -f environment.yml`
`conda activate darts`

### Datasets

Currently supported datasets are: CIFAR10, CIFAR100, and ImageNet (ILSVRC2012).

> To use a specific dataset when searching or training, you must pass the `--dataset cifar10/cifar100/imagenet` and `--data path/to/the/dataset` arguments.

### Run Search

`python train_search.py --batch_size 96 --pretrain_epochs 0 --init_channels 16 --amp --no_arch_metric`

> Default batch size is 72 (for low memory GPUs).

> Default dataset is CIFAR10.

> Logs and results will be saved in the `logs/search` folder.

### Run Search (DARTOpti)

`python train_search.py --batch_size 96 --arch_baseline ResNet18 --amp`

> Arguments for `--arch_baseline` can be: `ResNet18`, `ResNet50` or `Xception`.

> New architectures implemented in `genotypes.py` will automatically be available.

### Single Model Training

`python train.py --auxiliary --cutout --amp --auto_aug --arch D-DARTS_threshold_sparse_cifar10_0.85_50 --batch_size 128 --epoch 600 --init_channels 36`

> The genotype passed with `--arch` must be a `.txt` file stored in the `genotypes` folder.

> To use the automatic derivation algorithm presented in the paper, pass `--layers x` where `x` is an integer superior to the number of cells in the genotype. **/!\ Automatic derivation is not available when training a DARTOpti architecture (i.e., optimized from an existing architecture).**

> Logs and results will be saved in the `logs/eval` folder.

### Single Model Evaluation

```bash
python evaluate_model.py --arch ResNet18_cifar100_threshold_sparse_0.85 --model_path best_models/DO-2-ResNet18_ImageNet.pth.tar --init_channels 64
``` 

## Evaluation Results on CIFAR-10
### Comparison with Other State-of-the-art Results (CIFAR-10)
 
|  Model  | FLOPs  | Params  | Batch size  | lr | DP | Performance |
|---|---|---|---|---|---|---|
| DARTS_V2    | 522M   | 3.36 | 96  |  0.025   | 0.2  | 97.00* |
| PC-DARTS    | 558M   | 3.63 | 96  |  0.025   | 0.2  | 97.43* |
| PDARTS      | 532M   | 3.43 | 96  |  0.025   | 0.2  | 97.50* |
| FairDARTS-a | 373M   | 2.83 | 96  |  0.025   | 0.2  | 97.46* |   
| DD-1        | 259M   | 1.68 | 128 |  0.025   | 0.2  | 97.33  |
| DD-4        | 948M   | 6.28 | 128 |  0.025   | 0.2  | 97.75  |
| DO-ResNet18 | 1.2G   | 36.3 | 128 |  0.025   | 0.2  | 97.39  |
| DO-ResNet50 | 1.5G   | 71.2 | 128 |  0.025   | 0.2  | 97.20  |

*: Official result, as stated in the corresponding paper.

### Comparison with Other State-of-the-art Results (ImageNet)
 
|  Model  | FLOPs  | Params  | Batch size  | lr | DP | Performance | Searched On |
|---|---|---|---|---|---|---|---|
| DARTS_V2    | 574M   | 4.7  | 96  |  0.025   | 0.2  | 73.3*  | CIFAR-100 |
| PC-DARTS    | 586M   | 5.3  | 96  |  0.025   | 0.2  | 74.9*  | CIFAR-100 |
| PDARTS      | 577M   | 5.1  | 96  |  0.025   | 0.2  | 74.9*  | CIFAR-100 |
| FairDARTS-D | 440M   | 4.3  | 96  |  0.025   | 0.2  | 75.6*  | ImageNet  |  
| DD-7        | 828M   | 6.4  | 128 |  0.025   | 0.2  | 75.6   | ImageNet  |
| DO-ResNet18 | 8.6G   | 56.3 | 128 |  0.025   | 0.2  | 77.0   | CIFAR-100 |
| DO-ResNet18 | 10.0G  | 73.2 | 128 |  0.025   | 0.2  | 76.3   | CIFAR-100 |

*: Official result, as stated in the corresponding paper.
    
# Acknowledgement 

 **This code is based on the implementation of [DARTS](https://github.com/quark0/darts) and [FairDARTS](https://github.com/xiaomi-automl/FairDARTS).**
