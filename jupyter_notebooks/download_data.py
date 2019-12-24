import requests
import zipfile
import os
from tqdm import tqdm

wikiart_zip =  "http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip"
dir_path = "./data"
# f = "/home/mszmelcz/Downloads/K2.zip"
r = requests.get(wikiart_zip, stream=True)

zip_size = int(r.headers['Content-Length'].strip())
print("Zip size: {:.2f} GB".format(zip_size / 1073741824)) # Bytes / (Number of Bytes in Gigabyte)
block_size = 1024
with open("tmp.zip", "wb") as f:
    with tqdm(total=zip_size, unit="iB", unit_scale=True) as t:
        for data in r.iter_content(block_size):
            t.update(len(data)) 
            f.write(data) 
    with zipfile.ZipFile(f) as zf:
        try:
            os.mkdir(dir_path)
            print("Directory {} created".format(dir_path.split("/")[1]))
        except FileExistsError:
            print("Directory {} already exists".format(dir_path))
        for member in tqdm(zf.infolist(), desc="Extracting "):
    	    try:
    	        zf.extract(member, dir_path)
    	    except zipfile.error as e:
    	        print("Something went wrong {}".format(e))
        # zf.extractall(path=dir_path)
        print("File {} extrated into {}".format(wikiart_zip.split("/")[-1], dir_path.split("/")[1]))

