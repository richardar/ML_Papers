import kagglehub
from .extract_data import extractFile
import shutil
import os
def DownloadAndExtract():

    path = kagglehub.dataset_download("ayush1220/cifar10")

    print("Path to dataset files:", path)
    shutil.move(os.path.join(path,'cifar10'),os.getcwd())


    # extractFile(path)