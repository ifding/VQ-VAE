'''Author: Fei Ding
'''
import os
import time
import argparse
from tqdm import tqdm
import numpy as np
import warnings
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import segmentation_models_pytorch as smp
from PIL import Image
from dataset import SegCTDataset, TestDataset, get_training_trasnforms, get_valid_transforms
from utils import Meter, epoch_log, MixedLoss, training_plot
warnings.filterwarnings("ignore")

class Trainer(object):
    '''This class takes care of training and validation of our model'''
    def __init__(self, args, model, dataloaders):
        self.fold = 1
        self.total_folds = 5
        self.batch_size = {"train": args.batch_size, "val": int(args.batch_size//2)}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = args.lr
        self.num_epochs = args.epochs
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.criterion = MixedLoss(10.0, 2.0)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                              mode="min", 
                                                              patience=3, 
                                                              verbose=True)
        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        
        self.dataloaders = dataloaders
        self.save_dir = args.save_dir
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        
    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Epoch: {epoch} | Phase: {phase} | Time: {start}")
        #batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
#         tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1 ) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)
#             tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            val_loss = self.iterate(epoch, "val")
            self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, "{}/best_model.pth".format(self.save_dir))
            print()  


def train(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    train_dataset = SegCTDataset(
        txt='../datasets/train.txt', 
        images_dir=args.image_dir,
        masks_dir=args.mask_dir,
        augmentation=get_training_trasnforms('light'), 
    )

    valid_dataset = SegCTDataset(
        txt='../datasets/test.txt',
        images_dir=args.image_dir,
        masks_dir=args.mask_dir,        
        augmentation=get_valid_transforms(), 
    )
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              num_workers=8)
    valid_loader = DataLoader(valid_dataset, 
                              batch_size=int(args.batch_size//2), 
                              shuffle=False, 
                              num_workers=4) 
    dataloaders = {"train": train_loader, "val": valid_loader}
    
    model = smp.Unet(args.arch, 
                     encoder_weights=args.encoder_weights, 
                     activation=None)
    
    
    model_trainer = Trainer(args, model, dataloaders)
    model_trainer.start()
    
    losses = model_trainer.losses
    dice_scores = model_trainer.dice_scores # overall dice
    iou_scores = model_trainer.iou_scores
    
    training_plot(losses, args.save_dir, "losses")
    training_plot(dice_scores, args.save_dir, "dice_scores")
    training_plot(iou_scores, args.save_dir, "iou_scores")


def test(args):
    if not os.path.exists(args.newmask_dir):
        os.makedirs(args.newmask_dir)    
        
    test_dataset = TestDataset(txt='../datasets/test.txt', 
                               images_dir=args.image_dir,
                               size=320,
                               mean=[0.5, 0.5, 0.5],
                               std=[0.5, 0.5, 0.5])

    test_loader = DataLoader(test_dataset,batch_size=1,num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = smp.Unet(args.arch, 
                     encoder_weights=args.encoder_weights, 
                     activation=None).to(device)
    
    model.eval()
    state = torch.load('{}/best_model.pth'.format(args.save_dir), map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])    
    print("Model Loaded")
    
    print(test_dataset.images)

    for idx, batch in enumerate(tqdm(test_loader)):
        # Get the inputs and labels
        img = batch.to(device)
        with torch.no_grad():
            # Forward propagation
            preds = torch.sigmoid(model(img))[0]
            print(preds.size())
            preds = preds.squeeze(0).detach().cpu().numpy()*255 #.round()
            preds = preds.astype(np.uint8)
            preds = Image.fromarray(preds)
            preds.save(os.path.join(args.newmask_dir,test_dataset.images[idx]))
            exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CT Image Segmentation.')
    parser.add_argument('--arch', type=str, default='resnet34')  
    parser.add_argument('--encoder-weights', type=str, default='imagenet')     
    parser.add_argument('--image-dir', type=str, default='../datasets/2d_images/')
    parser.add_argument('--mask-dir', type=str, default='../datasets/2d_masks/')
    parser.add_argument('--save-dir', type=str, default='./save/exp')  
    parser.add_argument('--newmask-dir', type=str, default='./save/lungmask_gen')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs of training.')
    parser.add_argument('--batch-size', type=int,  default=8, help='batch-size')
    parser.add_argument('--start-epoch',type = int ,default=0, help='Start training epoch')
    parser.add_argument("--lr", type=float, default=0.0005, help='Learning rate')

    args = parser.parse_args()    
    
    train(args)
    #test(args)