from __future__ import absolute_import

import torch
import torchvision.transforms as T

from .import_torchreid import build_model

class ReIDFeatureExtractor(object):

    def __init__(
        self,
        model='resnet50',
        ckpt_file=None,
        gpu=-1,
        image_size=(256, 128),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        pixel_norm=True
    ):    

        if gpu >= 0:
            torch.backends.cudnn.benchmark = True
            self.device = torch.device('cuda:{}'.format(gpu))
        else:
            self.device = torch.device('cpu')

        self._model = build_model(
            model,
            num_classes=1,
            pretrained=not ckpt_file,
            use_gpu=gpu >= 0
        )

        if ckpt_file:
            ckpt = torch.load(ckpt_file, map_location=torch.device('cpu'))
            self._model.load_state_dict(ckpt["state_dict"])

        self._model.eval()
        self._model = self._model.to(self.device)

        # Build transform functions
        transforms = []
        transforms += [T.Resize(image_size)]
        transforms += [T.ToTensor()]
        if pixel_norm:
            transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
        preprocess = T.Compose(transforms)

        to_pil = T.ToPILImage()
        self.preprocess = preprocess
        self.to_pil = to_pil
        
    def __call__(self, img):
        image = self.to_pil(img)
        image = self.preprocess(image)
        images = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self._model(images)

        return features
