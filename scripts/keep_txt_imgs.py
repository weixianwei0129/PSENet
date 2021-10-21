import glob
import shutil
import os

all_txt = glob.glob("5/*.txt")

for txt in all_txt:
    basename = os.path.basename(txt).split('.')[0]
    img = txt.replace('.txt', '.jpg')
    shutil.copy(txt, f"train/{basename}.txt")
    shutil.copy(img, f"train/{basename}.jpg")

