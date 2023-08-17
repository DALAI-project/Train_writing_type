import torch
import argparse
import numpy as np
import random
import time
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset import CustomDataset
from utils import load_images_from_folder, initialize_model, calc_metrics

parser = argparse.ArgumentParser(description='Train function for binary classifier', 
    usage='%(prog)s [optional arguments]', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#
parser.add_argument('--epochs', action='store', type=int, required=True, help='Number of epochs. REQUIRED')
parser.add_argument('--run_name', action='store', type=str, required=True, help='Name for savefile/tensorboard. REQUIRED')
parser.add_argument('--lr', action='store', type=float, required=False, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--train_data', action='store', type=str, required=False, default='./data/train/', help='Path to training data folder')
#parser.add_argument('--test_data', action='store', type=str, required=False, default='/home/ubuntu/data/empty-classifier/test_data/data/', help='Path to test data folder')
parser.add_argument('--seed', action='store', type=int, required=False, default=42, help='Seed for splitting data')
parser.add_argument('--batch', action='store', type=int, required=False, default=16, help='Batch size')
parser.add_argument('--num_workers', action='store', type=int, required=False, default=1, help='Number of workers for loading data')
parser.add_argument('--weight_decay', action='store', type=float, required=False, default = 0.0001, help='weight decay for optimizer')
parser.add_argument('--mean_std', action='store', type=str, required=False, default = 'own', help='Tells, which means and stds to use in normalization. Options: own, ImageNet')
parser.add_argument('--label_smooth', action='store', type=float, required=False, default = 0.0, help='Use label smoothing loss')
parser.add_argument('--save_model_path', action='store', type=str, required=False, default='./runs/models', help='empty_classifier')
parser.add_argument('--limit_val', action='store', type=float, required=False, default = 2.0, help='precentage of calculating validation iterations')
parser.add_argument('--image_load_mode', action='store', type=str, required=False, default='shrink', help='Whether to use shrinking while loading images. Options: PIL, shrink')
parser.add_argument('--patience', action='store', type=int, required=False, default = 100, help='Stop training if no improvement after patience epochs')
parser.add_argument('--simple_transform', action='store', type=bool, required=False, default = False, help='Whether to use simple augmentation in training')
parser.add_argument('--own_transform', action='store', type=bool, required=False, default = False, help='Whether to use own augmentation in training')
parser.add_argument('--double_lr', action='store', type=bool, required=False, default = False, help='Whether to use different lrs for feature and classifier parts of the model')
parser.add_argument('--freeze', action='store', type=bool, required=False, default = False, help='Whether to use different lrs for feature and classifier parts of the model')
parser.add_argument('--model_path', action='store', type=str, required=False, default='', help='path to old model')
parser.add_argument('--img_size', action='store', type=int, required=False, default = 224, help='Image size used in training')
parser.add_argument('--bilinear', action='store', type=bool, required=False, default = False, help='Use bilinear resize. if false use bicubic')
parser.add_argument('--from_scratch', action='store', type=bool, required=False, default = False, help='Train from scratch')

args = parser.parse_args()

def train(args):
    print(args)
    # set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #make dataloaders
    train_valid_paths = load_images_from_folder(args.train_data)
    np.random.shuffle(train_valid_paths)
    train_set = CustomDataset(train_valid_paths[:int(0.9*len(train_valid_paths))], 
                              norm_mean_std = args.mean_std, 
                              simple_transformation=args.simple_transform,
                              own_transformation=args.own_transform,
                              image_load_mode = args.image_load_mode,
                              img_size = args.img_size,
                              bilinear=args.bilinear) 
    valid_set = CustomDataset(train_valid_paths[int(0.9*len(train_valid_paths)):], 
                              norm_mean_std = args.mean_std, 
                              simple_transformation=True, 
                              image_load_mode=args.image_load_mode,
                              img_size = args.img_size,
                              bilinear=args.bilinear)
    
    trainloader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, drop_last=True)
    validloader = DataLoader(valid_set, batch_size=args.batch, shuffle =True, num_workers=args.num_workers)
    
    # Create model etc.
    model = initialize_model(3,model_path=args.model_path, freeze = args.freeze, from_scratch=args.from_scratch)
    model = model.to(device)
    if args.double_lr:
        optimizer = torch.optim.AdamW([{"params": model.features.parameters(), "lr": args.lr/10},
                        {"params": model.classifier.parameters(), "lr": args.lr}
                        ], weight_decay= args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[args.lr/10, args.lr], steps_per_epoch=len(trainloader), epochs=args.epochs)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay= args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(trainloader), epochs=args.epochs)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    # TENSORBOARD LOGGER name
    writer = SummaryWriter('./runs/' + args.run_name)
    
    #CREATE A DIR FOR THE MODELS
    if not os.path.isdir(args.save_model_path):
        os.makedirs(args.save_model_path)
    
    #FOR LOOP FOR TRAINING limit valid, 
    since = time.time()
    best_loss = 10000.0
    best_fit = 0.0
    no_impro = 0
    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)
        
        model.train()
        running_loss =0.0
        running_corrects =0
        #TRAIN loop
        for inputs, labels in tqdm(trainloader):
            inputs = inputs.to(device)
            labels = labels.long().to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # Get model outputs and calculate loss
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(1), labels)

            # backward + optimize 
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            _, outputs = torch.max(outputs, 1)
            
            # statistics
            running_loss += loss.item() * inputs.size(0)
        
        # Calculate model losses
        epoch_loss = running_loss / (len(trainloader)*args.batch)
        # Log model losses to Tensorboard
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        if args.double_lr:
            writer.add_scalar("Learning_rate_feat", scheduler.get_last_lr()[0], epoch)
            writer.add_scalar("Learning_rate_cls", scheduler.get_last_lr()[1], epoch)
        else:
            writer.add_scalar("Learning_rate", scheduler.get_last_lr()[0], epoch)
            
        print('Train Loss: {:.4f}'.format(epoch_loss))

        #VALIDATION LOOP
        # zero the parameter gradients
        optimizer.zero_grad()
        model.eval()
        running_loss = 0.0
        iteration = 0
        all_outs = np.array([])
        all_labels = np.array([])
        for inputs, labels in tqdm(validloader):
            iteration += 1
            inputs = inputs.to(device)
            labels = labels.long().to(device)
            
            # forward with adaptive 
            with torch.no_grad():
                # Get model outputs and calculate loss
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(1), labels)
                #save all validation outputs and labels into an array for metrics calculations
                _, outputs = torch.max(outputs, 1)
                all_outs = np.append(all_outs, outputs.cpu().numpy())
                all_labels = np.append(all_labels, labels.cpu().numpy())
                    
            # statistics
            running_loss += loss.item() * inputs.size(0)
            if iteration==int(args.limit_val*len(validloader)):
                break
            
        # Calculate model losses
        epoch_loss = running_loss / (iteration*args.batch)
        # Log model metrics to Tensorboard
        valmetrics = calc_metrics(all_labels, all_outs)
        #loss
        writer.add_scalar("Loss/valid", epoch_loss, epoch)
        #accuracies
        writer.add_scalar("Accuracy/valid", valmetrics['acc'], epoch)
        writer.add_scalar("Accuracy/balanced_valid", valmetrics['acc_bal'], epoch)
        #precisions
        writer.add_scalar("Precision/kasi", valmetrics['prec_kasi'], epoch)
        writer.add_scalar("Precision/kone", valmetrics['prec_kone'], epoch)
        writer.add_scalar("Precision/yhd", valmetrics['prec_yhd'], epoch)
        #recalls
        writer.add_scalar("Recall/kasi", valmetrics['rec_kasi'], epoch)
        writer.add_scalar("Recall/kone", valmetrics['rec_kone'], epoch)
        writer.add_scalar("Recall/yhd", valmetrics['rec_yhd'], epoch)
        #f-scores
        writer.add_scalar("F-score/kasi", valmetrics['f_score_kasi'], epoch)
        writer.add_scalar("F-score/kone", valmetrics['f_score_kone'], epoch)
        writer.add_scalar("F-score/yhd", valmetrics['f_score_yhd'], epoch)
        writer.add_scalar("F-score/balanced", valmetrics['balanced_f_score'], epoch)
        #fitness
        fitness = 0.75 * valmetrics['balanced_f_score'] + 0.25 * valmetrics['acc_bal']
        writer.add_scalar("Fitness/fitness", fitness, epoch)

        print('Valid Loss: {:.4f} Balanced Acc: {:.4f}'.format(epoch_loss, valmetrics['acc_bal']))

        #Check for improvements
        no_impro +=1
        # Save if model is the best 
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_acc_vl = valmetrics['acc_bal']
            save_model_path = os.path.join(args.save_model_path, args.run_name + '_val_loss.pth')
            torch.save(model.state_dict(), save_model_path)
            no_impro = 0
        if fitness > best_fit:
            best_fit = fitness
            best_acc_fit = valmetrics['acc_bal']
            save_model_path = os.path.join(args.save_model_path, args.run_name + '_fitness.pth')
            torch.save(model.state_dict(), save_model_path)
            no_impro = 0
        if no_impro == args.patience:
            print("Early stopping on epoch ", epoch)
            break
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Balanced Val Acc on best model(val_loss): {:4f}'.format(best_acc_vl))
    print('Balanced Val Acc on best model(fitness): {:4f}'.format(best_acc_fit))
    

if __name__ == '__main__':
    train(args)