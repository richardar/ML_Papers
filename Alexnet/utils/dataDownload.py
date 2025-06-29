import kagglehub
from .extract_data import extractFile
import shutil
def DownloadAndExtract():

    path = kagglehub.dataset_download("ayush1220/cifar10")

    print("Path to dataset files:", path)


    # extractFile(path)