import os
from configparser import Interpolation

import torch
import torch.nn.functional as F
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader


from dataset import DepthDataset
from utils import visualize_img, ssim
from model import Net
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CyclicLR



class Solver():

    def __init__(self, args):
        # prepare a dataset
        self.args = args
        self.alpha = 0.6 #test1 = 0.5 test2= 0.3 test3 = 1 test4 = 0.7 test5 = 0.6


        if self.args.is_train:
            self.train_data = DepthDataset(train=DepthDataset.TRAIN,
                                           data_dir=args.data_dir,
                                           transform=transforms.Compose([
                                               transforms.Resize(224),
                                               transforms.CenterCrop(224)
                                           ]))
            self.val_data = DepthDataset(train=DepthDataset.VAL,
                                         data_dir=args.data_dir,
                                         transform=transforms.Compose([
                                               transforms.Resize(224),
                                               transforms.CenterCrop(224)
                                           ]))

            self.train_loader = DataLoader(dataset=self.train_data,
                                           batch_size=args.batch_size,
                                           num_workers=4,
                                           shuffle=True, drop_last=True)


            # turn on the CUDA if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #Model
            self.net = Net().to(self.device)

            self.criterion1 = torch.nn.MSELoss()
            self.criterion2 = torch.nn.L1Loss()
            self.optim = torch.optim.Adam(self.net.parameters() , lr=self.args.lr)
            self.scheduler = CyclicLR(self.optim,base_lr=0.001, max_lr=0.01, step_size_up=2000, step_size_down=2000,mode="triangular2")


            if not os.path.exists(args.ckpt_dir):
                os.makedirs(args.ckpt_dir)


        else:
            '''
            self.test_set = DepthDataset(train=DepthDataset.VAL,
                                    data_dir=self.args.data_dir,
                                    transform=transforms.Compose([
                                             transforms.Resize(224),
                                             transforms.CenterCrop(224)
                                         ]))
            ckpt_file = os.path.join("checkpoint", self.args.ckpt_file)
            self.net.load_state_dict(torch.load(ckpt_file, weights_only=True))
            '''

            # turn on the CUDA if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Model
            self.net = Net().to(self.device)

            self.test_set = DepthDataset(train=DepthDataset.VAL,
                                         data_dir=self.args.data_dir,
                                         transform=transforms.Compose([
                                             transforms.Resize(224),
                                             transforms.CenterCrop(224)
                                         ]))
            ckpt_file = os.path.join("checkpoint", self.args.ckpt_file)
            self.net.load_state_dict(torch.load(ckpt_file, weights_only=True))



    def fit(self):

        args = self.args

        for epoch in range(args.max_epochs):
            self.net.train()
            for step, inputs in enumerate(self.train_loader):
                rgb = inputs[0].to(self.device)
                #print("RGB input images size: ",rgb.size())


                depth = inputs[1].to(self.device)
                #print("Depth map label size: ", depth.size())


                pred = self.net(rgb)
                #print("Depth map predicted size: ", pred.size())

                #loss = self.alpha*self.criterion1(pred, depth) + (1-self.alpha)*self.criterion2(pred, depth)
                loss = self.alpha*torch.sqrt(self.criterion1(pred, depth)) + (1-self.alpha)*(1-ssim(pred, depth))


                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.scheduler.step()

                print("Epoch [{}/{}] Loss: {:.3f} ".
                      format(epoch + 1, args.max_epochs, loss.item()))


            if (epoch + 1) % args.evaluate_every == 0:
                #self.evaluate(DepthDataset.TRAIN)
                self.evaluate(DepthDataset.VAL)

                self.save(args.ckpt_dir, args.ckpt_name, epoch + 1)


    def evaluate(self, set):

        args = self.args
        if set == DepthDataset.TRAIN:
            dataset = self.train_data
            suffix = "TRAIN"
        elif set == DepthDataset.VAL:
            dataset = self.val_data
            suffix = "VALIDATION"
        else:
            raise ValueError("Invalid set value")

        loader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=4,
                            shuffle=False, drop_last=False)

        self.net.eval()
        ssim_acc = 0.0
        rmse_acc = 0.0
        with torch.no_grad():
            for i, (images, depth) in enumerate(loader):
                output = self.net(images.to(self.device))
                ssim_acc += ssim(output, depth.to(self.device)).item()
                rmse_acc += torch.sqrt(F.mse_loss(output, depth.to(self.device))).item()

                if i % self.args.visualize_every == 0:
                    visualize_img(images[0].cpu(),
                                  depth[0].cpu(),
                                  output[0].cpu().detach(),
                                  suffix=suffix)

        print("RMSE on", suffix, ":", rmse_acc / len(loader))
        print("SSIM on", suffix, ":", ssim_acc / len(loader))



    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)



    def test(self):

        loader = DataLoader(self.test_set,
                            batch_size=self.args.batch_size,
                            num_workers=4,
                            shuffle=False, drop_last=False)

        self.net.eval()
        ssim_acc = 0.0
        rmse_acc = 0.0
        with torch.no_grad():
            for i, (images, depth) in enumerate(loader):
                output = self.net(images.to(self.device))
                ssim_acc += ssim(output, depth.to(self.device)).item()
                rmse_acc += torch.sqrt(F.mse_loss(output, depth.to(self.device))).item()
                if i % self.args.visualize_every == 0:
                    visualize_img(images[0].cpu(),
                                  depth[0].cpu(),
                                  output[0].cpu().detach(),
                                  suffix="TEST")
        print("RMSE on TEST :", rmse_acc / len(loader))
        print("SSIM on TEST:", ssim_acc / len(loader))




