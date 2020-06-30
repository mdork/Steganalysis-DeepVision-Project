from tqdm import tqdm
import glob
import jpegio as jpio
from PIL import Image
import numpy as np
from scipy.fftpack import dct

## Write dictionary
img_path = '/export/data/mdorkenw/data/alaska2/'
folders = ['Cover', 'JUNIWARD', 'JMiPOD', 'UERD']
save_path = img_path + 'DCT/'

# for tech in folders:
for folder in folders:
    images = glob.glob(img_path + folder + "/*.jpg")

    for idx, img in tqdm(enumerate(images)):
        jpegStruct = jpio.read(img)
        dct_coeff = dct(np.array(Image.open(img)))
        # dct_coeff = np.stack(jpegStruct.coef_arrays, axis=2)
        # breakpoint()
        np.save(save_path + folder + '/' + str(idx).zfill(4) + '2.npy', dct_coeff)
