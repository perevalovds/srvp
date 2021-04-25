# Stochastic Latent Residual Video Prediction (SRVP)

Official implementation of the paper *Stochastic Latent Residual Video Prediction* (Jean-Yves Franceschi,* Edouard Delasalles,* Mickael Chen, Sylvain Lamprier, Patrick Gallinari), accepted and presented at ICML 2020.


## [Article](http://proceedings.mlr.press/v119/franceschi20a.html)


## [Presentation](https://icml.cc/virtual/2020/poster/5773)


## [Preprint](https://arxiv.org/abs/2002.09219)


## [Project Website](https://sites.google.com/view/srvp/)


## [Pretrained Models](https://data.lip6.fr/srvp/)


## Requirements

All models were trained with Python 3.7.6 and PyTorch 1.4.0 using CUDA 10.1.

A list required Python packages is available in the `requirements.txt` file.

To speed up training, we recommend to activate mixed-precision training in the options, whose performance gains were tested on the most recent Nvidia GPU architectures (starting from Volta).
We used Nvidia's [Apex](https://nvidia.github.io/apex/) (v0.1) in mixed-precision mode (`O1`) to produce results reported in the paper.
We also integrated PyTorch's more recent [mixed-precision training package](https://pytorch.org/docs/stable/amp.html) (made available in PyTorch 1.6.0), which should give similar results.
This is, however, an experimental feature and we cannot guarantee that it achieves the same results as Apex.


## Datasets

### Stochastic Moving MNIST

During training, this dataset is generated on the fly.
In order to generate a consistent testing set in an `.npz` file, the following commands should be executed:
```bash
python -m preprocessing.mmnist.make_test_set --data_dir $DIR --seq_len 25
```
for the stochastic version of the dataset, or
```bash
python -m preprocessing.mmnist.make_test_set --data_dir $DIR --deterministic --seq_len 100
```
for the deterministic version, where `$DIR` is the directory where the testing set should be saved.

### KTH

To download the dataset at a given path `$DIR`, execute the following command:
```bash
bash preprocessing/kth/download.sh $DIR
```
(see also [https://github.com/edenton/svg/blob/master/data/download_kth.sh](https://github.com/edenton/svg/blob/master/data/download_kth.sh) from the official implementation of [SVG](https://github.com/edenton/svg)).

In order to respectively train and test a model on this dataset, the following commands should be run:
```bash
python preprocessing/kth/convert.py --data_dir $DIR
```
and
```bash
python preprocessing/kth/make_test_set.py --data_dir $DIR
```

### Human3.6M

This dataset can be downloaded at [http://vision.imar.ro/human3.6m/description.php](http://vision.imar.ro/human3.6m/description.php), after obtaining access from its owners.
Videos for every subject are included in `.tgz` archives. Each of these archives should be extracted in the same folder.

To preprocess the dataset in order to use it for training and testing, these videos should be processed using the following command:
```bash
python preprocessing/human/convert.py --data_dir $DIR
```
where `$DIR` is the directory where Human3.6M videos are saved.

Finally, the testing set is created by choosing extracts from testing videos, with the following command:
```bash
python preprocessing/human/make_test_set.py --data_dir $DIR
```

All processed videos are saved in the same folder as the original dataset.

### BAIR

To download the dataset at a given path `$DIR`, execute the following command:
```bash
bash preprocessing/bair/download.sh $DIR
```
(see also [https://github.com/edenton/svg/blob/master/data/download_bair.sh](https://github.com/edenton/svg/blob/master/data/download_bair.sh) from the official implementation of [SVG](https://github.com/edenton/svg)).

In order to respectively train and test a model on this dataset, the following command should be run:
```bash
python preprocessing/bair/convert.py --data_dir $DIR
```


## Training

In order to launch training on multiple GPUs, launch the following command:
```bash
OMP_NUM_THREADS=$NUMWORKERS python -m torch.distributed.launch --nproc_per_node=$NBDEVICES train.py --device $DEVICE1 $DEVICE2 ...
```
followed by the training options, where `$NBDEVICES` is the number of GPUs to be used, `$NUMWORKERS` is the number of processes per GPU to use for data loading (should be equal to the value given to the option `n_workers`), and `$DEVICE1 $DEVICE2 ...` is a list of GPU indices whose length in equal to `$NBDEVICES`.
Training can be accelerated using options `--apex_amp` or `--torch_amp` (see [requirements](#Requirements)).

Data directory (`$DATA_DIR`) and saving path (`$SAVE_DIR`) must be given using options `--data_dir $DATA_DIR --save_path $SAVE_DIR`.

Training parameters are given by the following options:
- for Stochastic Moving MNIST:
```
--ny 20 --nz 20 --beta_z 2 --nt_cond 5 --nt_inf 5 --dataset smmnist --nc 1 --seq_len 15
```
- for Deterministic Moving MNIST:
```
--ny 20 --nz 20 --beta_z 2 --nt_cond 5 --nt_inf 5 --dataset smmnist --deterministic --nc 1 --seq_len 15 --lr_scheduling_burnin 800000 --lr_scheduling_n_iter 100000
```
- for KTH:
```
--ny 50 --nz 50 --n_euler_steps 2 --res_gain 1.2 --archi vgg --skipco --nt_cond 10 --nt_inf 3 --obs_scale 0.2 --batch_size 100 --dataset kth --nc 1 --seq_len 20 --lr_scheduling_burnin 150000 --lr_scheduling_n_iter 50000 --val_interval 5000 --seq_len_test 30
```
- for Human3.6M:
```
--ny 50 --nz 50 --n_euler_steps 2 --res_gain 1.2 --archi vgg --skipco --nt_cond 8 --nt_inf 3 --obs_scale 0.2 --batch_size 100 --dataset human --nc 3 --seq_len 16 --lr_scheduling_burnin 325000 --lr_scheduling_n_iter 25000 --val_interval 20000 --batch_size_test 8 --seq_len_test 53
```
- for BAIR:
```
--ny 50 --nz 50 --n_euler_steps 2 --archi vgg --skipco --nt_cond 2 --nt_inf 2 --obs_scale 0.71 --batch_size 192 --dataset bair --nc 3 --seq_len 12 --lr_scheduling_burnin 1000000 --lr_scheduling_n_iter 500000
```

Please also refer to the help message of `train.py`:
```bash
python train.py --help
```
which lists all options and hyperparameters to train SRVP models.



## Testing

To evaluate a trained model, the script `test.py` should be used as follows:
```bash
python test.py --data_dir $DATADIR --xp_dir $XPDIR --lpips_dir $LPIPSDIR
```
where `$XPDIR` is a directory containing a checkpoint and the corresponding `json` configuration file (see the pretrained models for an example), `$DATADIR` is the directory containing the test set, and `$LPIPSDIR` is a directory where [v0.1 LPIPS weights](https://github.com/richzhang/PerceptualSimilarity/tree/master/lpips/weights) (from the official repository of [*The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*](https://github.com/richzhang/PerceptualSimilarity)) are downloaded.

To run the evaluation on GPU, use the option `--device $DEVICE`.

Model file name can be specified using the option `--model_name $MODEL_NAME` (for instance, to load best models selected on the evaluation sets of KTH and Human3.6M: `--model_name model_best.pt`).

PSNR, SSIM and LPIPS results reported in the paper were obtained with the following options:
- for stochastic Moving MNIST:
```bash
python test.py --data_dir $DATADIR --xp_dir $XPDIR --lpips_dir $LPIPSDIR --nt_gen 25
```
- for deterministic Moving MNIST:
```bash
python test.py --data_dir $DATADIR --xp_dir $XPDIR --lpips_dir $LPIPSDIR --n_samples 1 --nt_gen 100
```
- for KTH:
```bash
python test.py --data_dir $DATADIR --xp_dir $XPDIR --lpips_dir $LPIPSDIR --nt_gen 40
```
- for Human3.6M:
```bash
python test.py --data_dir $DATADIR --xp_dir $XPDIR --lpips_dir $LPIPSDIR --nt_gen 53
```
- for BAIR:
```bash
python test.py --data_dir $DATADIR --xp_dir $XPDIR --lpips_dir $LPIPSDIR --nt_gen 30
```
Adding option `--fvd` additionally computes FVD.

Please also refer to the help message of `test.py`:
```bash
python test.py --help
```



## Troubleshooting

[It has been reported](https://github.com/edouardelasalles/srvp/issues/8) that using Apex mixed-precision training in specific configurations may lead to an excessive RAM usage due to [this memory leak issue in Apex](https://github.com/NVIDIA/apex/issues/634).
We refer to the links hereinabove for solutions to this problem.


Please feel free to create an issue for any other problem that you might encounter using our code.


## (Denis Perevalov) Damaging experiments

I explored how gradual network damaging affects video prediction on KTH videos.

1) You will see three experiments - resulted video and network image.

2) Each experiment contains 100 iterations, starting from original network and then network is randomly damaged again and again.
Videos contains repetitive frames, network "starts" from 10 frames, and you see here its prediction.
Start frames are mixed, but network changes - so you will see how prediction changes and degrades.

video 1: 40 frames per iteration
video 2: 120 frames per iteration
video 3: 21 frames per iteration

3) About damaging:
Iteration 1 - original network,
before iteration 2,3,...100 - we damage 100 random values of network.
The damaging is in multiplying given network weight on float random value from 0 to 1.
- It's really "Halzheimer" forgetting style, isn't it!

The place for the damage is chosen as random layer and random point there.
So, small layers are damaged more densely than huge layers.

3) About drawing network:
each image corresponds to each iteration,
I draw 1D and 2D network matrices "as is" and project 4D matrices to 2D using simple slicing.


