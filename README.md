# AutoMix
 Implementation for 'AutoMix: Mixup Networks for Sample Interpolation via Cooperative Barycenter Learning'

## Usage
```bash
usage: AutoMix.py [-h] [--method {baseline,bc,mixup,automix,adamixup}]
                  [--arch {mynet,resnet18}]
                  [--dataset {IMAGENET,CIFAR10,CIFAR100,MNIST,FASHION-MNIST,GTSRB,MIML}]
                  [--data_dir DATA_DIR] [--batch_size BATCH_SIZE] [--gpu GPU]
                  [--lr LR] [--num_workers NUM_WORKERS] [--momentum MOMENTUM]
                  [--weight_decay WEIGHT_DECAY] [--parallel PARALLEL]

AutoMix: Mixup Networks for Sample Interpolation via Cooperative Barycenter
Learning

optional arguments:
  -h, --help            show this help message and exit
  --method {baseline,bc,mixup,automix,adamixup}
                        Method : [baseline, bc, mixup, automix, adamixup]
  --arch {mynet,resnet18}
                        Backbone architecture : [mynet, resnet18]
  --dataset {IMAGENET,CIFAR10,CIFAR100,MNIST,FASHION-MNIST,GTSRB,MIML}
                        Dataset to be trained : [IMAGENET, CIFAR10, CIFAR100,
                        MNIST, FASHION-MNIST, GTSRB, MIML]
  --data_dir DATA_DIR   Path to the dataset
  --batch_size BATCH_SIZE
                        Batch_size for training
  --gpu GPU             GPU lists can be used
  --lr LR               Learning rate
  --num_workers NUM_WORKERS
                        Num of multiple threads
  --momentum MOMENTUM   Momentum for optimizer
  --weight_decay WEIGHT_DECAY
                        Weight_decay for optimizer
  --parallel PARALLEL   Train parallelly with multi-GPUs?

```
One can simply run the ```AutoMix.py``` to train ImageNet by
```bash
python AutoMix.py --method=baseline --arch=resnet18 --dataset=IMAGENET --data_dir=/media/reborn/Others2/ImageNet --batch_size=32 --lr=0.01 --gpu=0,1 --num_workers=8 --parallel=True --log_path=./automix.log
```

## Download the ImageNet dataset
The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) dataset has 1000 categories and 1.2 million images. 

1. Download the images from http://image-net.org/download-images

   1.1. Download **devkit** (including details of 1000 classes)

   ```bash
   wget http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_devkit_t12.tar.gz
   ```
   1.2. Download **train** dataset

   ```bash
   wget http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_train.tar
   ```
   1.3. Download **test** dataset
   ```bash
   wget http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_val.tar
   ```
   1.4. Download **validate** dataset
   ```bash
   wget http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_test_v10102019.tar
   ```

2. **Extract** the **training** data:
  ```bash
  mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
  tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
  find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
  cd ..
  ```

3. **Extract** the **validation** data and move images to subfolders:
  ```bash
  mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
  wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
  ```

## MNIST Experiments

- Baseline
```bash
python AutoMix.py --method=baseline --arch=mynet --dataset=MNIST --data_dir=Dataset --batch_size=100 --lr=0.1 --gpu=0 --num_workers=8 --parallel=True --epoch=100 --kfold=10
```

- BC [1]
```bash
python AutoMix.py --method=bc --arch=mynet --dataset=MNIST --data_dir=Dataset --batch_size=100 --lr=0.1 --gpu=0 --num_workers=8 --parallel=True --epoch=100 --kfold=10
```

- Mixup [2]
```bash
python AutoMix.py --method=mixup --arch=mynet --dataset=MNIST --data_dir=Dataset --batch_size=100 --lr=0.1 --gpu=0 --num_workers=8 --parallel=True --epoch=100 --kfold=10
```

- Manifold-Mixup [3]
```bash
python AutoMix.py --method=manifoldmixup --arch=mynet --dataset=MNIST --data_dir=Dataset --batch_size=100 --lr=0.1 --gpu=0 --num_workers=8 --parallel=True --epoch=100 --kfold=10
```

- AutoMix
```bash
python AutoMix.py --method=automix --arch=mynet --dataset=MNIST --data_dir=Dataset --batch_size=100 --lr=0.1 --gpu=0 --num_workers=8 --parallel=True --epoch=100 --kfold=10
```

## CIFAR10 Experiments

- Baseline
```bash
python AutoMix.py --method=baseline --arch=resnet18 --dataset=CIFAR10 --data_dir=Dataset/CIFAR10 --batch_size=100 --lr=0.1 --gpu=0 --num_workers=8 --parallel=True --epoch=300 --kfold=10
```

- BC [1]
```bash
python AutoMix.py --method=bc --arch=resnet18 --dataset=CIFAR10 --data_dir=Dataset/CIFAR10 --batch_size=100 --lr=0.1 --gpu=0 --num_workers=8 --parallel=True --epoch=300 --kfold=10
```

- Mixup [2]
```bash
python AutoMix.py --method=mixup --arch=resnet18 --dataset=CIFAR10 --data_dir=Dataset/CIFAR10 --batch_size=100 --lr=0.1 --gpu=0 --num_workers=8 --parallel=True --epoch=300 --kfold=10
```

- Manifold-Mixup [3]
```bash
python AutoMix.py --method=manifoldmixup --arch=resnet18 --dataset=CIFAR10 --data_dir=Dataset/CIFAR10 --batch_size=100 --lr=0.1 --gpu=0 --num_workers=8 --parallel=True --epoch=2000 --weight_decay=0.0001 --lr_schedule=500,1000,1500 --kfold=10
```

- AutoMix
```bash
python AutoMix.py --method=automix --arch=resnet18 --dataset=CIFAR10 --data_dir=Dataset/CIFAR10 --batch_size=100 --lr=0.1 --gpu=0 --num_workers=8 --parallel=True --epoch=300 --kfold=10
```

## CIFAR100 Experiments

- Baseline
```bash
python AutoMix.py --method=baseline --arch=resnet18 --dataset=CIFAR100 --data_dir=Dataset/CIFAR100 --batch_size=100 --lr=0.1 --gpu=0 --num_workers=8 --parallel=True --epoch=300 --kfold=10
```

- BC [1]
```bash
python AutoMix.py --method=bc --arch=resnet18 --dataset=CIFAR100 --data_dir=Dataset/CIFAR100 --batch_size=100 --lr=0.1 --gpu=0 --num_workers=8 --parallel=True --epoch=300 --kfold=10
```

- Mixup [2]
```bash
python AutoMix.py --method=mixup --arch=resnet18 --dataset=CIFAR100 --data_dir=Dataset/CIFAR100 --batch_size=100 --lr=0.1 --gpu=0 --num_workers=8 --parallel=True --epoch=300 --kfold=10
```

- Manifold-Mixup [3]
```bash
python AutoMix.py --method=manifoldmixup --arch=resnet18 --dataset=CIFAR100 --data_dir=Dataset/CIFAR100 --batch_size=1024 --lr=0.1 --gpu=0 --num_workers=8 --parallel=True --epoch=2000 --weight_decay=0.0001 --lr_schedule=500,1000,1500 --kfold=10
```

- AutoMix
```bash
python AutoMix.py --method=automix --arch=resnet18 --dataset=CIFAR100 --data_dir=Dataset/CIFAR100 --batch_size=100 --lr=0.1 --gpu=0 --num_workers=8 --parallel=True --epoch=300 --kfold=10
```

## Reference

[1] Tokozume Y , Ushiku Y , Harada T . Between-class Learning for Image Classification[J]. 2017.
[2] Zhang H , Cisse M , Dauphin Y N , et al. mixup: Beyond Empirical Risk Minimization[J]. 2017.
[3] Verma V , Lamb A , Beckham C , et al. Manifold Mixup: Better Representations by Interpolating Hidden States[J]. 2018.