# Steganalysis - DeepVision Project
Deep Vision project SS 2020 by S. Gruber and M. Dorkenwald 

### Steganalysis 
The goal in steganalysis is to classify/detect if an image contains a hidden message or not. Therefore it is typically used in espionage, thus important for law enforcement. The problem itself is inherently difficult as the cover image is not provided during inference. We use a diverse dataset from kaggle which was acquired with various cameras, jpeg compression and steganography algorithms. Here some examples:
![alt text](https://github.com/mdork/Steganalysis-DeepVision-Project/blob/master/scripts/visualization/diff_img/img13.png?raw=true)
More examples can be found in scripts/visualization/. You can see a strong correlation between the difference image in the DCT space and in the RGB space. Moreoever, the difference are mostly hidden in the high frequency parts. The amount of pixel differences correlates with the used JPEG compression (see the jupyter notebook in scripts for more details). We achieved the best results with a pretrained efficientnet b0 (trained on 12 classes different JPEG compression) with a weighted AUC score on the kaggle test set of 0.91. 

### Setup



- Clone repository into preferred directory



    ```
    git clone https://github.com/mdork/Steganalysis-DeepVision-Project.git
    ```



- Create virtual conda environment



    ```
    cd Steganalysis-DeepVision-Project/
    conda create -n myenv --file requirements.txt
    ```



- Download data from kaggle 



    ```
    kaggle competitions download -c alaska2-image-steganalysis 
    ```

- Run python scripts after changing the directories to extract JPEG compression and DCT coefficients. 

    ```
    python scripts/JPEG_compression.py
    python scripts/DCT_coefficients.py
    ```
- For training a model you only have to change the network_base_setup.txt file. The least thing you've to do is to change the data path.
