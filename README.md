# Writing Type Training

Code for training a deep learning model to detect the writing type of document images. 

## Intro

The component can be used for classifying input files to 'machinewritten' and 'handwritten' document and documents containing both writing types.

The following input file formats are accepted: 

- .jpg
- .png 
- .tiff 

## Setup

### Creating virtual environment using Anaconda

The component can be safely installed and used in its own virtual environment. 
For example an Anaconda environment 'writing_type' can be created with the command

### Installing dependencies in LINUX (Tested with Ubuntu 20.04)

```conda create -n writing_type python=3.7```

Now the virtual environment can be activated by typing

```conda activate writing_type```

Install required dependencies/libraries by typing 

```
pip install -r requirements.txt
```

The latter command must be executed in the folder where the requirements.txt file is located.

NB! If you are having problems with the above command, installing this library can help. `conda install -c conda-forge poppler`

## Training

Before training the data should be organized into a train folder, where there are two folders: an empty folder, containing empty images, and a ok folder for non-empty images. An example of the organization is depicted below.

```
├──writing_type_training
      ├──models
      ├──data
      |   ├──train
      |       ├──combination
      |       ├──hand_written
      |       └──type_written
      ├──runs
      ├──train.py
      ├──utils.py
      ├──dataset.py
      ├──constants.py
      └──requirements.txt
```

An example of training command. More arguments can be found in the train.py file or by running `python train.py -h`. 

```python train.py --epochs 10 --run_name 'run1'```

By default, the code uses a pretrained Imagenet model. If you want to train from scratch you can specify it in the training command as below. 

```python train.py --epochs 10 --run_name 'run1' --from_scratch True```

If you want to use data augmentations (i.e. rotate, colorjitter, sharpness, blur, affine and erasing), you can specify it in the training command as below.

```python train.py --epochs 10 --run_name 'run1' --own_transform True```

If you want to use different learning rates for base model and classification head, you can specify it in the training command as below.

```python train.py --epochs 10 --run_name 'run1' --double_lr True```

If you want to use freeze the base model during training, you can specify it in the training command as below.

```python train.py --epochs 10 --run_name 'run1' --freeze True```

If you want to change learning rate, number of workers or batch size, you can specify it in the training command as below.

```python train.py --epochs 10 --run_name 'run1' --lr 0.01 --num_workers 4 --batch 8```

All of the examples above can be used in one command.

### Visualizing training metrics with Tensorboard

You can visualize training metrics with Tensorboard by running a following command in the training folder.

```tensorboard --logdir runs```

After this you can in your preferred browser go to this link http://localhost:6006/. There you can see how your training is progressing. 

### Information about the saved the models

The models are saved into `./runs/models` folder.

The code saves two models based on different metrics. The first model is saved based on a "fitness" score that is calculated `0.75 * balanced f1 score + 0.25 * balanced_accuracy`. The second model is saved based on validation loss.
