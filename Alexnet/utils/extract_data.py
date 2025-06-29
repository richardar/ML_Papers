import zipfile
import os


def extractFile(path):
    with zipfile.ZipFile(os.path.join(path,'cifar10.zip'),'r') as zip_ref:
        zip_ref.extractall(os.getcwd())