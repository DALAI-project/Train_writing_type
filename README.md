# writing_type_classifierV3
Writing type classifier made in plain pytorch instead of fastai

## Intro

The component can be used for classifying input files to 'machinewritten' and 'handwritten' document and documents containing both.

The following input file formats are accepted: 

- .pdf
- .jpg
- .png 
- .tiff 

If the input is a .pdf file, it is transformed into an image file before further processing.
Each page of a multipage .pdf file is reported as a separate document in the output file.

The results classified by the component is saved as a .csv file, where each row corresponds
to a single input file (with the exception of multipage .pdf files). The columns of the output
file contain the following data:

- Filename of the input file ('filename')
- Predicted class ('writing_type_class')
- Confidence of prediction

## Setup

### Creating virtual environment using Anaconda

The component can be safely installed and used in its own virtual environment. 
For example an Anaconda environment 'empty_env' can be created with the command

### Installing dependencies in LINUX (Tested with Ubuntu 20.04)

`conda create -n writing_type python=3.7`

Now the virtual environment can be activated by typing

`conda activate writing_type`

Install required dependencies/libraries by typing 

```
conda install -c conda-forge poppler
pip install -r requirements.txt
```

The latter command must be executed in the folder where the requirements.txt file is located.

## Running the component in LINUX

The filepath of the input folder and the path for saving the output .csv file are given
as arguments when running the component using command-line.

For example, if the 'test.py' code file is saved in folder
`/home/<username>/writing_type_classifierV3/`, the input files are located in folder `/home/<username>/writing_type_classifierV3/input/` and model file is 
located in folder `/home/<username>/writing_type_classifierV3/models/` as 'writing_type_v1.onnx'
and the output file is saved in the file `/home/<username>/writing_type_classifierV3/results/results.csv`, the component can
be run by executing the command

`python3 test.py --data_path /home/<username>/writing_type_classifierV3/input --model_file_name writing_type_v1.onnx --results_file_path /home/<username>/writing_type_classifierV3/results/results.csv`

in the folder `/home/<username>/writing_type_classifierV3/`.

If you don't want to specify the paths as arguments, please save your input data to the folder called 'input' and run the command
python3 test.py
By default, the empty_content_results.csv will be saved to the same path to the folder called 'results'.

For example, if the 'test.py' code file is saved in folder
`/home/<username>/writing_type_classifierV3/`, the input files have to be located in folder `/home/<username>/writing_type_classifierV3/input/`
the component can be run by executing the command

`python3 empty_classifier.py`

and the output file will be saved to the path `/home/<username>/writing_type_classifierV3/results/writing_type_results.csv`.
