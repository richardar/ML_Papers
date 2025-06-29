
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pathlib
import os
import torch


# train_dir = os.path.join(os.getcwd(),'cifar10','train')
# test_dir  = os.path.join(os.getcwd(),'cifar10','test')


# class_names = [name for name in os.listdir(test_dir)]
# class_names.sort()
# print(class_names)

# class_dict = {name:num for num,name in enumerate(class_names)}
# print(class_dict)


class cifarDS(Dataset):

    def __init__(self,path:str,transform:transforms = None):
        self.path = path
        self.transform = transform
        self.all_images = list(pathlib.Path(self.path).glob('*/*.png'))
        self.class_names = [name for name in os.listdir(self.path)]
        self.class_names.sort()
        print(self.class_names)

        self.class_dict = {name:num for num,name in enumerate(self.class_names)}
        print(self.class_dict)


    def __len__(self):
        return len(self.all_images)
    
    def getImage(self,path:str) -> Image.Image: 

        return Image.open(path)

    def __getitem__(self, index:str ) -> tuple[torch.Tensor,int]:

        path = self.all_images[index]

        image = self.getImage(path)
        classdir = os.path.basename(os.path.dirname(path))
        classname = self.class_dict[classdir]

        # y_onehot = torch.nn.functional.one_hot(torch.tensor(classname), num_classes=10).float()

        if self.transform:
            return self.transform(image) ,classname
        

        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

            return self.transform(image), classname