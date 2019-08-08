import os.path
import random
from data.base_dataset import BaseDataset, get_transform
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image
import torch


class AlignedDataset(BaseDataset):
    """A dataset class for 3D paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.volume_size = opt.volume_size
        self.dir_AB = opt.dataroot # get parent paths of A and B
        # self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        self.A_path = os.path.join(self.dir_AB, 'image')
        self.B_path = os.path.join(self.dir_AB, 'mask')
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        # we manually crop and flip in __getitem__ to make sure we apply the same crop and flip for image A and B
        # we disable the cropping and flipping in the function get_transform
        self.transform_A = get_transform(opt, grayscale=(input_nc == 1), crop=False, flip=False)
        self.transform_B = get_transform(opt, grayscale=(output_nc == 1), crop=False, flip=False)
        self.patient_list = self.get_patient_list(opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Return a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read 3D images given a random integer index
        A_3d = []
        B_3d = []
        # patient_list = self.get_patient_list()
        patient_id = self.patient_list[index].zfill(4)
        A_path_3d = os.path.join(self.A_path, patient_id)
        B_path_3d = os.path.join(self.B_path, patient_id)
        for i in range(self.volume_size):
            A = Image.open(os.path.join(A_path_3d, str(i+1)+'.png'))
            B = Image.open(os.path.join(B_path_3d, str(i+1)+'.png'))
            A = A.resize((self.opt.load_size, self.opt.load_size), Image.BICUBIC)
            B = B.resize((self.opt.load_size, self.opt.load_size), Image.BICUBIC)
            # apply the same cropping to both A and B
            if 'crop' in self.opt.preprocess:
                x, y, h, w = transforms.RandomCrop.get_params(A, output_size=[self.opt.crop_size, self.opt.crop_size])
                A = A.crop((x, y, w, h))
                B = B.crop((x, y, w, h))
            # apply the same flipping to both A and B
            if (not self.opt.no_flip) and random.random() < 0.5:
                A = A.transpose(Image.FLIP_LEFT_RIGHT)
                B = B.transpose(Image.FLIP_LEFT_RIGHT)
            # call standard transformation function
            A = self.transform_A(A)
            # A = torch.unsqueeze(A, dim=0)
            A_3d.append(A)
            B = self.transform_B(B)
            # B = torch.unsqueeze(B, dim=0)
            B_3d.append(B)
        A_3d = torch.cat(A_3d, 0)
        A_3d = torch.unsqueeze(A_3d, dim=0)
        B_3d = torch.cat(B_3d, 0)
        B_3d = torch.unsqueeze(B_3d, dim=0)
        return {'A': A_3d, 'B': B_3d, 'A_paths': A_path_3d, 'B_paths': B_path_3d}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.patient_list)

    @staticmethod
    def get_patient_list(opt):
        # patient_list = []
        patient_name_list = open(opt.phase + '.txt').read().splitlines()
        # cimg_name_list = [patient_name.split(' ')[0] for patient_name in patient_name_list]
        return patient_name_list
