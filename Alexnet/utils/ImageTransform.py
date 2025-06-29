
from torchvision import transforms

ImageTransform = transforms.Compose([
    transforms.Resize((224,224)),

    transforms.ToTensor()
])