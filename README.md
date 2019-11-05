# VQ-VAE

### Using pip install the following packages:

python 3.7

tensorflow==1.15.0

tensorflow-gpu==1.15.0

tensorflow-probability==0.7.0

dm-sonnet==1.35


### Training

```
python main.py --train True
```

### Load saved model and run on Test data

best_timestamp is like 2019_11_05_15_21_08,  can be found in `logs` or `checkpoint_dir`

```
python main.py --timestamp <best_timestamp>
```

It also save the zq to `logs` folder, `cifar_latent.npy`, to read it just by `np.load`