# MPANet-for-IR
A learning-based method for image restoration

## Data preparation

For training data, you can download the datasets from the official url.  Then generate training patches for training by:

```shell
python3 generate_patches_SIDD.py
```

## Training

##### Denoising

To train MPANet for image denoising, you can begin the training by:

```shell
python train_denoising.py
```

##### Super resolution

To train MPANet for image SR, you can begin the training by:

```shell
python train_super_resolution.py
```

##### Deblurring

To train MPANet for image deblurring, you can begin the training by:

```shell
python train_deblurring.py
```

##### Deraining

To train MPANet for image deraining, you can begin the training by:

```shell
python train_deraining.py
```

## Evaluation

To evaluate MPANet, you can run the test code according to the file name to get the test results on different datasets:

```shell
python test_***.py
```

