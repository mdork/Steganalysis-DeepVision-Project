from tqdm import tqdm
import glob
import numpy as np
import jpegio as jpio

## Write dictionary
img_path = '/export/data/mdorkenw/data/alaska2/'
folders = ['Cover', 'JUNIWARD', 'JMiPOD', 'UERD']

img_paths = []
jpeg_comp = []

# for tech in folders:

images = glob.glob(img_path + "Cover/*.jpg")

for img in tqdm(images):
    jpegStruct = jpio.read(img)

    if (jpegStruct.quant_tables[0][0, 0] == 2):
        # print(img, 'Quality Factor is 95')
        jpeg_comp.append(2)
    elif (jpegStruct.quant_tables[0][0, 0] == 3):
        # print(img, 'Quality Factor is 90')
        jpeg_comp.append(1)
    elif (jpegStruct.quant_tables[0][0, 0] == 8):
        # print(img, 'Quality Factor is 75')
        jpeg_comp.append(0)

assert len(images) == len(jpeg_comp)
np.save(img_path + 'JPEG_compression.npy', np.array(jpeg_comp))