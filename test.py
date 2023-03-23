import onnxruntime as ort
from tqdm import tqdm
import pandas as pd
import os
import time
from pdf2image import convert_from_path
import torchvision.transforms as t
from torchvision.transforms.functional import InterpolationMode
import PIL
import numpy as np
import argparse
import sys

# Skriptin tavoitteena on luokitella läjä kuvia sen mukaan, ovatko ne tyhjiä vai sisällöllisiä.
# Skripti tuottaa lopputuloksena csv-tiedoston, joka sisältää kaksi columnia "filename" ja "empty_content"
# Filename on alkuperäisen kuvan tiedostonimi ja empty_content kertoo onko kuva tyhjä "empty"
# vai sisällöllinen "content".

parser = argparse.ArgumentParser('argument for writing type classifier with new data')

#Määrittele polku, mistä aineisto löytyy, jos se ei ole "input"
parser.add_argument('--data_path', type=str, default="./input/",
                    help='path to data')
#Määrittele mallitiedoston nimi, jos se on muuttunut
parser.add_argument('--model_file_name', type=str, default="writing_type_v1.onnx",
                    help='name of model onnx file')
#Määrittele polku, mistä mallitiedosto löytyy, jos se ei ole "models"
parser.add_argument('--model_file_path', type=str, default="./models/",
                    help='path to model onnx file')
#Määrittele polku, minne csv-tiedosto tallennetaan, jos se ei ole "results"
parser.add_argument('--results_file_path', type=str, default="./results/writing_type_results_" + time.strftime("%Y%m%d%M") + ".csv",
                    help='path to save results as a csv file')

args = parser.parse_args()
print(sys.argv[1:])

transformation = t.Compose([
                            t.Resize((224,224), interpolation=InterpolationMode.BICUBIC),
                            t.ToTensor(),
                            t.Normalize([0.882, 0.883, 0.899], [0.088, 0.089, 0.094])
                        ])

model = ort.InferenceSession(args.model_file_path + args.model_file_name)
class_names ={0:"Käsinkirjoitettu",1:"Konekirjoitettu", 2:"Yhdistelmä"}

# Function that returns a list of the filepaths in the input data folder
def get_filepaths(path):
    files_list = []
    other_filetypes = False
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.pdf')):
                filepath = os.path.join(root, name)
                files_list.append(filepath)
            else:
                other_filetypes = True
    if other_filetypes == True:
        print('Sovellus hyväksyy vain .png, .pdf, .jpg ja .tiff -tiedostotyypit.')
    return files_list

def process_input(path):
    # Get filepaths for input files
    #file_paths = get_filepaths(args.input_path)
    file_paths = get_filepaths(path)
    pdf = False
    for i, filepath in enumerate(file_paths):
        # Get filename and file extension
        name, extension = os.path.splitext(filepath)
        # Pdf input files are transformed into image format
        if extension == '.pdf':
            pdf = True
            images = convert_from_path(filepath)
            # For multi-page pdf files each page is processed as a separate image
            for j, img in enumerate(images):
                if len(images) > 1:
                    # Add page number to each image extracted from multi-page pdf
                    new_name = name + '_' + str(j)
                    filepath = new_name
                else:
                    filepath = name
                # Save pdf page as a .jpg image
                img.save(filepath + '.jpg', 'JPEG')
    if pdf == True:
        print('Pdf-tiedostot muutetaan jpg-kuvatiedostoiksi ja tallennetaan takaisin kansioon.\
        Monisivuisesta pdf-tiedostosta jokainen sivu tallennetaan erillisenä kuvatiedostona.')
        
def softmax(x):
    return(np.exp(x)/np.exp(x).sum())
        
def classes_to_csv(path):
    df=[]
    df = pd.DataFrame(df)
    img_list= []
    classes = []
    confs = []
    for root, dirs, files in os.walk(path):
        for file_name in tqdm(files):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                #load image & transform into correct type for model
                img_path = os.path.join(root, file_name)
                img = PIL.Image.open(img_path)
                img.draft('RGB',(224,224))
                img = transformation(img.convert('RGB'))
                img = img.unsqueeze(0).detach().cpu().numpy()
                
                #run through model
                input = {model.get_inputs()[0].name: img}
                res = model.run(None, input)
                
                #get results i.e. class and confidence 
                ind = np.argmax(res[0], 1).item()
                conf = res[0][0]
                class1 = class_names[ind]
                img_filename = os.path.basename(img_path)
                img_list.append(img_filename)
                classes.append(class1)
                confs.append(softmax(conf)[ind])

    df['filename'] = img_list
    df['writing_type_class'] = classes
    df['confidence'] = confs
    return df
    
    
def main():
    process_input(args.data_path)
    df = classes_to_csv(args.data_path)
    if not os.path.exists(args.results_file_path.split('/')[-2]):
        os.makedirs(args.results_file_path.split('/')[-2])
    df.to_csv(args.results_file_path, index=None)
    num_kasi = len(df['writing_type_class'][df['writing_type_class']=='Käsinkirjoitettu'])
    num_kone = len(df['writing_type_class'][df['writing_type_class']=='Konekirjoitettu'])
    num_yhd = len(df['writing_type_class'][df['writing_type_class']=='Yhdistelmä'])
    print('Konekirjoitettuja löytyi', num_kone,'käsinkirjoitettuja löytyi', num_kasi, 'ja asiakirjoja missä on kumpaakin löytyi', num_yhd,'kpl')
    print('Kirjoitustyypin tunnistus suoritettu.')
    
    print('.csv-tiedosto tallennettu polkuun ' + args.results_file_path)
    
main()

sys.stderr.close()
sys.stderr = sys.__stderr__