'''
MEAN (list(float)): Means for normalization
STD (list(float)): Standard deviations for normalization
NUM_IMAGES (list(ints)): Sizes of validation, test and train sets
TRAIN_CLASS_NUM (list): List of class sizes (benign, cancer) in train set
VALID_CLASS_NUM (list): List of class sizes (benign, cancer) in validation set
BETA (float): Beta value for class_balanced loss
'''

MEAN = {'ImageNet': [0.485, 0.456, 0.406],
    'own' : [0.882, 0.883, 0.899]
       }

STD = {'ImageNet': [0.229, 0.224, 0.225],
    'own' : [0.088, 0.089, 0.094]
      }

BETA = 0.9999
