import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
sys.path.append(os.path.abspath("/raid/home/himanshus/experiments/repos/stable-diffusion/blip"))

import argparse, os, sys, glob
import PIL
import torch

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid, save_image
from torch import autocast
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader,Dataset
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything
# import robustbench
# from robustbench.data import load_cifar10, load_imagenet
# from robustbench.utils import load_model, clean_accuracy

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from scripts.custom_Dataset import CustomDataset
from IPython.display import display, clear_output
from custom_pgd import PGD
from blip.models.blip import blip_decoder

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean).view(-1, 1, 1)
        self.std = torch.Tensor(std).view(-1, 1, 1)

    def forward(self, x):
        if x.is_cuda:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
        return (x - self.mean) / self.std
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# from PIL import Image
# import requests
# import torch
# import torchvision
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms.functional import InterpolationMode
# import matplotlib.pyplot as plt
# import numpy as np
# import foolbox as fb
# from robustbench.data import load_cifar10
class parser():
    def __init__(self):
        # self.prompt = "a painting of a virus monster playing guitar"
        # self.init_img = "outputs/img2img-samples/grid-0003.png" #path to the input image
        # self.outdir = "../BLIP/experiment_1/imagenet/clean_test_generated_2048" #dir to write results to
        self.skip_grid = True
        self.skip_save = False
        self.ddim_steps = 50 #number of ddim sampling steps
        self.plms = False
        self.fixed_code = False
        self.ddim_eta = 0.0 #ddim eta (eta=0.0 corresponds to deterministic sampling
        self.n_iter = 1
        self.C = 4 #latent channels
        self.f = 8
        self.n_samples = 32 #batch size
        self.n_rows = 0
        self.scale = 5.0 #unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
        self.strength = 0.5 #strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image
        self.from_file = ""
        self.config = "configs/stable-diffusion/v1-inference.yaml"
        self.ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"
        self.seed = 42
        self.precision = "autocast" #["full", "autocast"]
        
def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.
from blip.models.blip import blip_decoder

image_size = 384
# image = load_demo_image(image_size=image_size, device=device)

model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
model_path = "./blip/checkpoints/model_base_capfilt_large.pth"
    
blip_model = blip_decoder(pretrained=model_path, image_size=image_size, vit='base')
blip_model.eval()
blip_model = blip_model.to(device)
opt = parser()
seed_everything(opt.seed)

config = OmegaConf.load(f"{opt.config}")
model = load_model_from_config(config, f"{opt.ckpt}")

model = model.to(device)
#added by himanshu for adding batch processing
blip_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
# classifier_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
classifier_transform = transforms.Compose([transforms.ToTensor()])
                                # transforms.Resize((96,96)),])
                                # classifier_normalize])
transform_blip = transforms.Compose([transforms.Resize(size=(384,384)),
                                blip_normalize
                                ])
# classifier_inv_normalize = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
#                                                      std = [ 1/0.229, 1/0.224, 1/0.225 ]),
#                                 transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
#                                                      std = [ 1., 1., 1. ]),
#                                ])
blip_inv_normalize = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1./0.26862954, 1./0.26130258, 1./0.27577711 ]),
                                transforms.Normalize(mean = [ -0.48145466, -0.4578275, -0.40821073 ],
                                                     std = [ 1., 1., 1. ]),])
                                # transforms.Resize((32,32))
                            #    ])
inverse_resize = transforms.Compose([transforms.Resize((32,32))])
# inverse_resize_2 = transforms.Compose([transforms.Resize((32,32))])
batch_size = opt.n_samples
num_classes = 100
# image_folder = '../BLIP/experiment_1/cifar10/clean_test/'
# label_file = '../BLIP/experiment_1/cifar10/clean_test_labels.txt'
# image_folder = '../BLIP/experiment_1/imagenet/clean_test_2048'
# label_file = '../BLIP/experiment_1/imagenet/clean_test_labels_2048.txt'
# captions_file = "../BLIP/experiment_1/cifar10/clean_test_captions.txt"
# test_dataset = CustomDataset(image_folder, label_file, transform=classifier_transform)
# test_dataset = datasets.CIFAR100("../data/", train=False, transform=classifier_transform, download=True)
# train_dataset = datasets.STL10(root="../data",
#                            split="train",
#                            transform=transform,
#                           download=True)
# test_dataset = datasets.STL10(root="../data",
#                          split="test",
#                          transform=transform)

# train_loader = torch.utils.data.DataLoader(train_dataset,
#                                        batch_size=batch_size,
#                                        shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
#                                      shuffle=False, num_workers=8)

# class_names = train_dataset.classes
# data_iterator = iter(train_loader)
# batch = next(data_iterator)
if opt.plms:
    raise NotImplementedError("PLMS sampler not (yet) supported")
    sampler = PLMSSampler(model)
else:
    sampler = DDIMSampler(model)

# os.makedirs(opt.outdir, exist_ok=True)
# outpath = opt.outdir

batch_size = opt.n_samples
n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
t_enc = int(opt.strength * opt.ddim_steps)
print(f"target t_enc is {t_enc} steps")

precision_scope = autocast if opt.precision == "autocast" else nullcontext
mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]
def inverse_normalize(batch_tensor, mean=mean, std=std):
    for tensor, m, s in zip(batch_tensor, mean, std):
        tensor.mul_(s).add_(m)
    return batch_tensor

def defense(img):
    x_samples = None
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                # clear_output(wait=True)
                init_image = transform_blip(img).to(device)
                # data = [list(batch[2])]
                # print(data)
                data = [blip_model.generate(init_image, sample=False, num_beams=3, max_length=20, min_length=5)]
                # print(data)
                num = len(img)
                init_image = inverse_normalize(init_image)
                init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
                sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
                for n in range(opt.n_iter):
                    for prompts in data:
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(num * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        # encode (scaled latent)
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*num).to(device))
                        # decode it
                        samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                unconditional_conditioning=uc,)

                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                toc = time.time()
    return inverse_resize(x_samples)
# from tqdm import tqdm

# mean_rgb = (0.4914, 0.4822, 0.4465)
# std_rgb = (0.2023, 0.1994, 0.2010)
# jitter_param = 0.4

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=mean_rgb,
#         std=std_rgb),
# ])

transform = transforms.Compose([
    transforms.ToTensor()])

# class Normalize(nn.Module):
#     def __init__(self, mean, std):
#         super(Normalize, self).__init__()
#         self.mean = torch.Tensor(mean).view(-1, 1, 1)
#         self.std = torch.Tensor(std).view(-1, 1, 1)

#     def forward(self, x):
#         if x.is_cuda:
#             self.mean = self.mean.to(x.device)
#             self.std = self.std.to(x.device)
#         return (x - self.mean) / self.std

# def calculate_accuracy(model, dataloader):
#     total_samples = 0
#     correct_predictions = 0
#     model = model.to(device)
#     model.eval()
#     with torch.no_grad():
#         for images, labels in tqdm(dataloader):
#             images = images.to(device)
#             labels = labels.to(device)

#             outputs = model(images)
#             _, predicted = torch.max(outputs, dim=1)

#             total_samples += labels.size(0)
#             correct_predictions += (predicted == labels).sum().item()

#     accuracy = correct_predictions / total_samples
#     return accuracy
test_dataset = datasets.CIFAR100(root="../data",train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=opt.n_samples, num_workers=8)
# from pytorchcv.model_provider import get_model as ptcv_get_model
# # import torch
# # from torch.autograd import Variable


    
# clf = ptcv_get_model("wrn28_10_cifar100", pretrained=True)
# norm = Normalize(mean_rgb, std_rgb)   
# clf = nn.Sequential(norm, clf)
# clf.eval()
# calculate_accuracy(clf, test_loader)
# import torchattacks
# atk = torchattacks.PGD(clf, eps=8./255., alpha=2./255.)
# atk.set_normalization_used(mean=mean_rgb, std=std_rgb)
# clf = clf.to(device)
# clf.eval()
# correct = 0
# total = 0

# for images, labels in tqdm(test_loader):
#     images, labels = images.to(device), labels.to(device)
#     adv_images = atk(images, labels)
#     p_img = defense(adv_images)
#     outputs = clf(defense(p_img))
#     _, predicted = torch.max(outputs.data, 1)
    
#     total += labels.size(0)
#     correct += (predicted == labels).sum().item()

# print(f'Robust Accuracy: {(100 * correct / total):.2f}%')



count = 0
labels_list = []
for sample in tqdm(test_loader):
    # adv_images = atk(sample[0], sample[1])
    # adv_images = classifier_inv_normalize(adv_images) 
    labels_list.extend(sample[1])  
    generated_image = defense(sample[0])
    for a_i in generated_image:
        
        save_image(a_i, "../BLIP/experiment_1/cifar100/clean_test_generated/{}.png".format(count))
        # ig_purified = defense(torch.unsqueeze(ig, dim=0))
        
        # save_image(ig_purified, "../BLIP/experiment_1/stl_10/adv_test_generated_e2e/cleaned/{}.png".format(count))
        count+=1

with open("../BLIP/experiment_1/cifar100/clean_test_labels.txt", 'w') as f:
    for integer in labels_list:
        f.write(f"{integer.item()}\n")