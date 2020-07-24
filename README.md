# Steganalysis - DeepVision Project
Deep Vision project SS 2020 by S. Gruber and M. Dorkenwald \
Info about Steganalysis



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
