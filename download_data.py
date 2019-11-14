import requests
import zipfile
import os

file_path = "/home/stanislaw/Downloads/Humpback-Whale-Identification-master.zip"
dir_path = "./data"

with zipfile.ZipFile(file_path) as zf:
    try:
        os.mkdir(dir_path)
        print("Directory {} created".format(dir_path.split("/")[1]))
    except FileExistsError:
        print("Directory {} already exists".format(dir_path))
    zf.extractall(path=dir_path)
    print("File {} extrated into {}".format(file_path.split("/")[-1], dir_path.split("/")[1]))

