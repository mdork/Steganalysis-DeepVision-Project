# Steganalysis - DeepVision Project
Deep Vision project SS 2020 by S. Gruber and M. Dorkenwald 

### Steganalysis 
The task in steganalysis is to classify/detect if an image contains a hidden message or not. Typically used in espionage, thus important for law enforcement. Thus a low rate of false positives is desired. The problem itself is inherently difficult as cover image is not provided during inference. We use a diverse dataset from kaggle which was acquired with various cameras, jpeg compression and steganography algorithms. Here some examples:


### Setup



- Clone repository into preferred directory



    ```
    git clone https://github.com/mdork/Steganalysis-DeepVision-Project.git
    ```



- Create virtual conda environment



    ```
    cd Steganalysis-DeepVision-Project/
    conda create -n myenv --file package-list.txt
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
