import requests
import zipfile
import os

wikiart_zip =  "http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip"
# file_path = "/home/stanislaw/Downloads/Humpback-Whale-Identification-master.zip"
dir_path = "./data"

r = requests.get(wikiart_zip)

with zipfile.ZipFile(r.content) as zf:
    try:
        os.mkdir(dir_path)
        print("Directory {} created".format(dir_path.split("/")[1]))
    except FileExistsError:
        print("Directory {} already exists".format(dir_path))
    zf.extractall(path=dir_path)
    print("File {} extrated into {}".format(wikiart_zip.split("/")[-1], dir_path.split("/")[1]))

