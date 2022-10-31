import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Iterable, Dict
import kornia.augmentation as K
import torch.nn.functional as F

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

def compose_augmentations(args_list, config):
    """
    Uses the args list to compose a single augmentation function.
    """
    augs_funcs = []

    for aug_name in args_list.split('-'):
        if aug_name == "color_jitter":
            augs_funcs.append(K.ColorJitter(
                brightness=config.COLOR_JITTER.brightness,
                contrast=config.COLOR_JITTER.contrast,
                saturation=config.COLOR_JITTER.saturation,
                hue=config.COLOR_JITTER.hue,
                p=config.COLOR_JITTER.color_p,
                same_on_batch=config.same_on_batch
            ))
        elif aug_name == "grayscale":
            augs_funcs.append(K.RandomGrayscale(
                p=config.GRAYSCALE.gray_p,
                same_on_batch=config.same_on_batch
            ))
        elif aug_name == "random_resized_crop":
            augs_funcs.append(K.RandomResizedCrop(
                size=(config.RANDOM_RESIZED_CROP.crop_size, config.RANDOM_RESIZED_CROP.crop_size), 
                scale=(config.RANDOM_RESIZED_CROP.crop_scale, 1.0), 
                ratio=(1.0 / config.RANDOM_RESIZED_CROP.crop_ratio, config.RANDOM_RESIZED_CROP.crop_ratio), 
                p=config.RANDOM_RESIZED_CROP.crop_p,
                same_on_batch=config.same_on_batch
            ))
        elif aug_name == "rotation":
            augs_funcs.append(K.RandomRotation(
                degrees=config.RANDOM_ROTATION.rotation_degrees, 
                p=config.RANDOM_ROTATION.rotation_p,
                same_on_batch=config.same_on_batch
            ))
        elif aug_name == "translate":
            augs_funcs.append(nn.ReplicationPad2d(config.TRANSLATE.pad))
            augs_funcs.append(K.RandomCrop(
                (config.TRANSLATE.crop_size, config.TRANSLATE.crop_size), 
                p=config.TRANSLATE.crop_p,
                same_on_batch=config.same_on_batch
            ))
        elif aug_name == "translate_v2":
            augs_funcs.append(RandomShiftsAug(config.TRANSLATE.pad))
        elif aug_name == "":
            pass
        else:
            raise Exception(f"Augmentation {aug_name} not found")

        obs_transform = nn.Sequential(*augs_funcs).cuda()

    return obs_transform

if __name__ == '__main__':
    import PIL.Image as Image
    from torchvision.transforms import ToPILImage, ToTensor
    # load png image using file path
    img = Image.open('examples/002_033_TURN_RIGHT_source.png')
    img = ToTensor()(img)
    
    img = img.unsqueeze(0)
    print(img.shape)

    class config:
        RANDOM_RESIZED_CROP = type('', (), {})()
        RANDOM_RESIZED_CROP.crop_size = 256
        RANDOM_RESIZED_CROP.crop_scale = 0.8
        RANDOM_RESIZED_CROP.crop_ratio = 1.0
        RANDOM_RESIZED_CROP.crop_p = 1.0
        RANDOM_ROTATION = type('', (), {})()
        RANDOM_ROTATION.rotation_degrees = 10
        RANDOM_ROTATION.rotation_p = 1.0
        GRAYSCALE = type('', (), {})()
        GRAYSCALE.gray_p = 1.0
        COLOR_JITTER = type('', (), {})()
        COLOR_JITTER.brightness = 0.1
        COLOR_JITTER.contrast = 0.1
        COLOR_JITTER.saturation = 0.1
        COLOR_JITTER.hue = 0.1
        COLOR_JITTER.color_p = 1.0
        TRANSLATE = type('', (), {})()
        TRANSLATE.pad = 100
        TRANSLATE.crop_p = 1.0
        TRANSLATE.crop_size = 256
        same_on_batch = False

    aug_name = "random_resized_crop-translate"
    aug_name = "random_resized_crop-translate_v2"
    augs = compose_augmentations(aug_name, config)

    img_aug = augs(img)

    img_aug = ToPILImage()(img_aug.squeeze())
    img_aug.save('examples/img1_' + aug_name + '.png')
