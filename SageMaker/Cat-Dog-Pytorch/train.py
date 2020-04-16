from __future__ import print_function, division
import sagemaker_containers
import logging
import sys
import PIL
import argparse
import json
import os
import torch
import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from collections import OrderedDict
import boto3
import tarfile
import datetime
import math
import copy


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
validation_loss=[]
min_val_loss=9999
train_accuracy=[]
validation_accuracy=[]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size
        
def _init_weights(model):
    if isinstance(model, nn.Conv2d):
        torch.nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            model.bias.data.fill_(0.01)
    elif isinstance(model, nn.BatchNorm2d):
        model.weight.data.uniform_()
        if model.bias is not None:
            model.bias.data.zero_()        
        
def _get_model(model_def,NUM_CLASSES,model_location):
    
    pretrained_resent=True
    if model_def == 'resnet18':
        model_ft = models.resnet18(pretrained=pretrained_resent)
    elif model_def == 'resnet34':
        model_ft = models.resnet34(pretrained=pretrained_resent)
    elif model_def == 'resnet50':
        model_ft = models.resnet50(pretrained=pretrained_resent)
    elif model_def == 'resnet101':
        model_ft = models.resnet101(pretrained=pretrained_resent)
    elif model_def == 'densenet121':
        model_ft = models.densenet121(pretrained=pretrained_resent)
    else:
        raise ValueError('Choose valid model def...')

    if model_def == 'resnet18' or model_def == 'resnet34' or model_def == 'resnet50' or model_def == 'resnet101':
        model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(OrderedDict([
                                            ('fc1', nn.Linear(num_ftrs, 256)),
                                            ('relu', nn.ReLU()),
                                            ('dropout', nn.Dropout(0.2)),
                                            ('fc2', nn.Linear(256, NUM_CLASSES))
                                            ]))
    # The classifier part is a single fully-connected layer (classifier): Linear(in_features=1024, out_features=1000).
    # we substitute it by or nn.Sequential 
                                    
    if  model_def == 'densenet121':
        model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Sequential(OrderedDict([
                                            ('fc1', nn.Linear(num_ftrs, 256)),
                                            ('relu', nn.ReLU()),
                                            ('dropout', nn.Dropout(0.2)),
                                            ('fc2', nn.Linear(256, NUM_CLASSES))
                                            ]))
        
    return model_ft

def sgdr_lr(max_lr, min_lr, cycle_length, current_step):
    current_step = current_step % cycle_length
    max_min_dist = max_lr - min_lr
    cos_arg = ((math.pi / 2) / cycle_length) * (current_step - 1)
    dist_penalty = 1 - math.cos(cos_arg)
    new_lr = max_lr - (max_min_dist * dist_penalty)
    return new_lr

def transfer_learning(args):
    '''Training function'''
    min_val_loss=9999
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("Device Type: {}".format(device))
    
    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # Create data loader objects from our helper functions
    logger.info('Training data set curation start:')
    logger.info(datetime.datetime.now())
    
    train_loader = Get_train_data_loader(args.batch_size, args.data_dir+'/train').get_data_loader()
    valid_loader = Get_validation_data_loader(args.batch_size, args.data_dir+'/validation').get_data_loader()
    test_loader = Get_test_data_loader(args.batch_size, args.data_dir+'/test').get_data_loader()

   
    logger.info('Training data set curation end:')
    logger.info(datetime.datetime.now())
    
    logger.debug("Processes {}/{} ({:.0f}%) of train data".format(
        len(train_loader.sampler), len(train_loader.dataset),
        100. * len(train_loader.sampler) / len(train_loader.dataset)
    ))

    # Load model and send to device
    model = _get_model(args.model_def,args.no_of_classes,args.model_location)
    if args.model_location is None:
        if is_distributed and use_cuda:
            # multi-machine multi-gpu case
            model = torch.nn.parallel.DistributedDataParallel(model)
        else:
            # single-machine multi-gpu case or single-machine or multi-machine cpu case
            model = torch.nn.DataParallel(model)
        
    model = model.to(device)
    
    logger.info("Model on cuda: {}".format(next(model.parameters()).is_cuda))    
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Optimize using SGD, notice every parameter is being trained!
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    # Learning rate scheduler, cutting lr by 10 every arg.lr_step
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)
    scheduler = None
    
    iters = 0
    for epoch in range(1, args.epochs + 1):     
        #Training
        logger.info('Epoch {}/{}'.format(epoch, args.epochs))
        logger.info('-' * 10)
        model, epoch_loss, epoch_acc, iters = train(train_loader, model, device, scheduler, optimizer, iters, args.lr, epoch,
                                             criterion, is_distributed, use_cuda)
        
        train_accuracy.append(epoch_acc)
        
         # log the information of the epoch
        logger.info('Train accuracy: {}'.format(train_accuracy))
        logger.info('Train Loss: {:.4f}'.format(epoch_loss))
        logger.info('Train Acc: {:.4f}'.format(epoch_acc))

        # Validation step
        epoch_loss, epoch_acc, wrong_pred_paths = validate(model, valid_loader, args.no_of_classes, args.batch_size, device)
        
        validation_accuracy.append(epoch_acc)
        validation_loss.append(epoch_loss)
        
        # log validation information
        logger.info('Validation accuracy: {}'.format(validation_accuracy))
        logger.info('Val Loss: {:.4f}'.format(epoch_loss))
        logger.info('Val Acc: {:.4f}'.format(epoch_acc))
        logger.info('')
        # Log learning rate
        for param_group in optimizer.param_groups:
            logger.info('Learning rate: {:.4f}'.format(param_group['lr']))
          
        # log wrong prediction paths
        #logger.info('Incorrect prediction paths: {}'.format(wrong_pred_paths))

        if epoch_loss<min_val_loss:
            min_val_loss=epoch_loss
            logger.info('storing best model Loss: {:.4f}'.format(epoch_loss))
            best_model=copy.deepcopy(model)
        
        # This at least needs 4 epochs
        if len(validation_loss)>4:
            logger.info('total iterations: {:.4f}'.format(iters))
            previous_loss=validation_loss[-4]
            min_loss=min(validation_loss[-3:])
            max_loss=max(validation_loss[-3:])
            
            if previous_loss<min_loss or abs(max_loss-min_loss)<0.01:
                logger.info('Validation loss is increasing, training more doesnt increase accuracy and causes overfitting')
                break
           
    print('Finished Training\n\n')
    
    print('Start Testing')
    test(best_model, test_loader, args.no_of_classes, args.batch_size, device, criterion, use_cuda)
    print('Finished Testing\n')
    
    # Save model
    save_model(best_model, args.model_dir)
    
def train(train_loader, model, device, scheduler, optimizer, iters, lr, epoch, criterion, is_distributed, use_cuda):
    
        # make sure model in training mode (batch norm only in training!)
        model.train()
        
        # step scheduler
        #scheduler.step()
        
        # init statistics we care about tracking
        running_loss = 0.0
        running_corrects = 0
        
        # batch loop, we loop over train_loader object
        for batch_idx, (data, target, paths) in enumerate(train_loader, 1):
            iters += 1
            
            update_lr = sgdr_lr(lr / (2 ** (epoch // 10)), .001, 100, iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = update_lr     
            
            data, target = data.to(device), target.to(device)
           
            # zero out gradients
            optimizer.zero_grad()
                 
            # get outputs of model on current data and calculate prediction
            output = model(data)
            
            _, preds = torch.max(output, 1)
            
            # calculate the CrossEntropyLoss for output and backpropagate error
            loss = criterion(output, target)
            
            loss.backward()
            
            if is_distributed and not use_cuda:
                # average gradients manually for multi-machine cpu case only
                _average_gradients(model)
                
            optimizer.step()
            
            # statistics
            running_loss += loss.item() * data.size(0)
            running_corrects += torch.sum(preds == target.data)
        
        # find the epoch's loss and acc using running numbers divided by data set size
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        return model, epoch_loss, epoch_acc, iters

def validate(model, valid_loader, NUM_CLASSES,batch_size, device):
    '''Vaidation function'''
    
    # set model to eval mode
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    wrong_pred_paths = []
   
    with torch.no_grad():
        for data, target, paths in valid_loader:
            
            data, target = data.to(device), target.to(device)
                    
            output = model(data)
            
            _, preds = torch.max(output, 1)
            loss = nn.CrossEntropyLoss()(output, target)
                        
            running_loss += loss.item() * data.size(0)
            running_corrects += torch.sum(preds == target.data)
            
            # logging incorrect prediction file names
            for ind, num in enumerate((preds == target.data)):
                if num == 0:
                    filepath = paths[ind]
                    wrong_pred_paths.append((filepath, output[ind].cpu().numpy()))
    
    # get epoch stats by dividing running stats by data size
    epoch_loss = running_loss / len(valid_loader.dataset)
    epoch_acc = running_corrects.double() / len(valid_loader.dataset)
    
    return epoch_loss,epoch_acc,wrong_pred_paths

def test(model, test_loader, NUM_CLASSES, batch_size, device, criterion, use_cuda):
    # track test loss
    test_loss = 0.0
    test_correct = 0.0
    wrong_pred_paths = []
    class_correct = list(0. for i in range(NUM_CLASSES))
    class_total = list(0. for i in range(NUM_CLASSES))
    model.eval()
    with torch.no_grad():
        for data, target, paths in test_loader:            
            data, target = data.to(device), target.to(device)                    
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update test loss
            test_loss += loss.item()*data.size(0)
            # convert output probabilities to predicted class
            _, preds = torch.max(output, 1)
            # compare predictions to true label
            correct_tensor = preds.eq(target.data.view_as(preds))
            test_correct = np.squeeze(correct_tensor.numpy()) if not use_cuda else np.squeeze(correct_tensor.cpu().numpy())
            for i in range(len(data)):
                label = target.data[i]
                class_correct[label] += test_correct[i].item()
                class_total[label] += 1
        
        # average test loss
        test_loss = test_loss/len(test_loader.dataset)
        logger.info('Test Loss: {:.6f}'.format(test_loss))
        for i in range(NUM_CLASSES):
            if class_total[i] > 0:
                logger.info('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    classes[i], 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                logger.info('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

        logger.info('Test Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))   

def save_model(model, model_dir):
    logger.info("Saving the model to {}.".format(model_dir))
    path = os.path.join(model_dir, 'model.pth')
    # save full model - model.module will be the method to predict image
    #torch.save(model, path)
    # save the part inside parallel data 
    torch.save(model.cpu().module, path)

#
#------------------------------------ Data Loader ---------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

MEAN = [0.413, 0.413, 0.413]
STD = [0.159, 0.161, 0.158]


class Make_weights_for_balanced_classes():
    
    def __init__(self,images,nclasses):
        self.images = images
        self.nclasses = nclasses

    def get_weights(self):
        count = [0] * self.nclasses                                                      
        for item in self.images:                                                         
            count[item[1]] += 1  # item is (img-data, label-id)
        weight_per_class = [0.] * self.nclasses                                      
        N = float(sum(count))  # total number of images                  
        for i in range(self.nclasses):                                                   
            weight_per_class[i] = N/float(count[i])                                 
        weight = [0] * len(self.images)                                              
        for idx, val in enumerate(self.images):                                          
            weight[idx] = weight_per_class[val[1]]     

        return weight
   
 
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class Get_train_data_loader():
    
    def __init__(self, batch_size,training_dir):
        self.batch_size = batch_size
        self.training_dir = training_dir

    def get_data_loader(self):
        
        logger.info("Get train data loader")

        data_transform = transforms.Compose([
            # randomly change the brightness, contrast and saturation
            # factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness] and the same for others
            # but hue (float or tuple of float (min, max)): How much to jitter hue.
            # hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            # Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.1),
            
            # The transform RandomResizedCrop crops the input image by a random size(within a scale range of 0.8 to 1.0
            # of the original size and a random aspect ratio in the default range of 0.75 to 1.33 ).
            #The crop is then resized to 256×256            
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])

        dataset = ImageFolderWithPaths(self.training_dir, data_transform)
        print(dataset)
        print(dataset.classes)

        weights = Make_weights_for_balanced_classes(
                    dataset.imgs, len(dataset.classes)).get_weights()

        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, sampler = sampler)
    
    
class Get_test_data_loader():
    
    def __init__(self, test_batch_size,testing_dir):
        self.batch_size = test_batch_size
        self.testing_dir = testing_dir

    def get_data_loader(self):
        
        logger.info("Get test data loader")
        data_transform = transforms.Compose([
            
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])

        dataset = ImageFolderWithPaths(self.testing_dir, data_transform)

        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)


class Get_validation_data_loader():
    
    def __init__(self, validation_batch_size,validation_dir):
        self.batch_size = validation_batch_size
        self.validation_dir = validation_dir

    def get_data_loader(self):
        
        logger.info("Get validation data loader")
        data_transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])

        dataset = ImageFolderWithPaths(self.validation_dir, data_transform)

        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)



    
    



#------------------------------ Run --------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--model_def', type=str, default='resnet18', metavar='S',
                        help='the base resnet model to use resnet(18, 34, 50)')
    
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    
    parser.add_argument('--lr_step', type=float, default=50,
                        help='learning rate scheduler step size (default: 50)')
    
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--data_location', type=str, default=None,
                        help='The location where the data is located')
    parser.add_argument('--model_location', type=str, default=None,
                        help='The location where the model is located')
    
    parser.add_argument('--mean', type=list, default=[0.325,0.315,0.130], metavar='ME',
                        help='Avg image mean (default: [0.325,0.315,0.130])')
    
    parser.add_argument('--std', type=list, default=[0.104, 0.134, 0.153], metavar='ST',
                        help='Std of images (default: [0.104, 0.134, 0.153])')
    
    parser.add_argument('--no_of_classes', type=int, default=2, metavar='C',
                        help='No of images events (default: 2)')
    
    parser.add_argument('--seed', type=int, default=1, metavar='SE',
                        help='random seed (default: 1)')
    
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

    env = sagemaker_containers.training_env()
    logger.info('Training started:')
    logger.info(datetime.datetime.now())
    logger.info(env.channel_input_dirs.get('training'))
    
    inbuilt=env.channel_input_dirs.get('training')
    # Container environment
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args=parser.parse_args()
    model_data = {}  
    classes=[]
    for dirpath, dirnames, filenames in os.walk(env.channel_input_dirs.get('training')+'/train'):
        classes=classes+dirnames
    logger.info(classes)  
    classes.sort()
        
    model_data['BaseModel']=args.model_def
          
    model_dir_location=os.environ['SM_MODULE_DIR']
    model_artifacts=model_dir_location.split('/')
    
    resource = boto3.resource('s3')
    my_bucket = resource.Bucket(model_artifacts[2]) 
    my_bucket.download_file(('/').join(model_artifacts[3:]),'/opt/ml/model/'+model_artifacts[-1])
    
    transfer_learning(args)