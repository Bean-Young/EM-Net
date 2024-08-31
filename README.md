# EM-Net

This project was created by Yuezhe Yang for the paper "**EM-Net: Effective and Morphology-aware Network for Skin Lesion Segmentation**" (paper link).

## ***Abstract***
Dermoscopic images are essential for diagnosing various skin diseases, as they enable physicians to observe subepidermal structures, dermal papillae, and deeper tissues otherwise invisible to the naked eye. However, segmenting lesions in these
images is challenging due to their irregular boundaries and the significant variability in lesion characteristics. To address these challenges, we propose a high-precision model that utilizes a hybrid feature extractor combining CNN and ViT architectures. This extractor captures both spatial and local information effectively.  Our model includes a boundary delineation component that uses a non-convex optimization function for learning general representations and accurately delineates lesion boundaries, thus enhancing the extraction of details. We integrate these texture
features with other raw features, enriching the feature information and preserving high resolution details in shallow feature maps. Additionally, we introduce a domain adaptive adversarial learning strategy to improve the model's generalization across different datasets. This strategy involves a discriminator that distinguishes between samples from varied datasets, optimizing the source domain distribution to enhance adaptability. We validated our model on multiple publicly available dermoscopic image datasets, such as ISIC, PH², PAD-UFES-20, and the University of Waterloo skin cancer database. The results confirm that our method achieves state-of-the-art
performance and demonstrates robust generalization capabilities.


## ***Prepare Data***

To validate the effectiveness of our model, we conducted extensive experiments on four skin lesion image datasets from different sources.

The details of the datasets used and their specific links are provided here for easy access to the data.

### **ISIC**

*ISIC*: The International Skin Imaging Collaboration (ISIC) is an international academia and industry partnership designed to reduce skin cancer morbidity and mortality through the development and use of digital skin imaging applications.

![ISIC](/demo/ISIC_0000006.jpg)

[ISIC 2016 Task1](https://challenge.isic-archive.com/data/): It includes 900 images for training and 379 images for testing.

[ISIC2017 Task1](https://challenge.isic-archive.com/data/#2017): It includes 2000 images for training and 600 images for testing.We did not use validation data.

[ISIC2018 Task1](https://challenge.isic-archive.com/data/#2018): It includes 2594 images for training and 1000 images for testing.We did not use validation data.

### **PH<sup>2</sup>**

*PH<sup>2</sup>*: The PH² dataset has been developed for research and benchmarking purposes, in order to facilitate comparative studies on both segmentation and classification algorithms of dermoscopic images. PH² is a dermoscopic image database acquired at the Dermatology Service of Hospital Pedro Hispano, Matosinhos, Portugal.

![PH<sup>2</sup>](/demo/IMD010.bmp)

[PH<sup>2</sup> Dataset](https://www.fc.up.pt/addi/ph2%20database.html): We used ten of these images for domain generalization and the remaining 190 images to test the performance of the model trained on ISIC2016.

### **PAD-UFES-20**

*PAD-UFES-20*: A skin lesion dataset composed of patient data and clinical images collected from smartphones.

![PAD-UFES-20](/demo/PAT_32_44_211.png)

[PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1): This dataset did not provide segmentation masks, so under the guidance of professional clinicians, we selected 30 representative skin lesion images to create the masks. The mask images are publicly available in a [*branch*](https://github.com/Bean-Young/EM-Net/tree/GroudTruth-for-PAD) of this project. We used two of these images for domain generalization, and the remaining 28 images were used to test the performance of the model trained on ISIC2016.

### **University of Waterloo skin cancer database**

*University of Waterloo skin cancer database*: The dataset is maintained by VISION AND IMAGE PROCESSING LAB, University of Waterloo. The images of the dataset were extracted from the public databases [*DermIS*](https://www.dermis.net/dermisroot/en/home/index.htm) and [*DermQuest*](https://www.emailmeform.com/builder/form/Ne0j8da9bb7U4h6t1f), along with manual segmentations of the lesions.

![University of Waterloo skin cancer database](/demo/46_orig.jpg)

[University of Waterloo](https://uwaterloo.ca/vision-image-processing-lab/research-demos/skin-cancer-detection): The dataset provided by the University of Waterloo includes complete mask labels and contains a total of 206 images. We used six of these images for domain generalization, and the remaining 200 images were used to test the performance of the model trained on ISIC2016.


## ***Set Up***

### Pytorch 2.0 (CUDA 11.8)
Our experimental platform is configured with 4 RTX 3090 GPUs (CUDA 11.8), and the code runs in a PyTorch 2.0 environment.

For details on the environment, please refer to the [`requirements.txt`](requirements.txt) file.

**Run the installation command:**
```
pip install -r requirements.txt
```
### Pretrain

You need to download Google's pre-trained ViT model, which can be obtained through [*this link*](https://console.cloud.google.com/storage/browser/_details/vit_models/imagenet21k/R50%2BViT-B_16.npz;tab=live_object): R50-ViT-B_16.

## ***Data Preprocessing***

Data preprocessing is divided into two stages: 
1) We convert the data into a uniform format 
2) We generate a list of all the data to be used.

To better adapt the input for deep learning networks and ensure optimal network performance, we need to preprocess the data. We have provided a packaged preprocessing and Morphology-aware Module(MM). You only need to run [`Prepare_data.py`](Prepare_data.py), or use the following commands to preprocess the files. The data will be uniformly converted into NPZ format, with each file containing the original unprocessed images, images obtained through the MM module, and binarized mask images.

```
python Prepare_data.py
```
We have briefly demcionstrated the images obtained from the MM Module.
![MM](/demo/MM.png)


To obtain the list of data used, you just need to run [`lists.py`](lists.py). This will generate the corresponding lists, including `train.txt`, `test_vol.txt`, and `all.lst` files.

```
python lists.py
```

## ***Train***

Once you have completed the data preprocessing, you can use the [`train.py`](train.py) file to train your own model.

>**N.B.** If you encounter any issues while running the code, please contact me.
```
python train.py
```

## ***Validation***

After completing the training, you can simply run [`test.py`](test.py) to perform testing on the test set. We have configured all the necessary parameters for testing.

However, please note that the pre-trained weights provided in the uploaded code of this project are only for demonstration purposes and should not be used directly for testing. We provided weights trained on the ISIC dataset, which you can apply for extensive validation on skin lesion image datasets.

>If you would like the pre-trained model for a specific dataset, please email me at *<wa2214014@stu.ahu.edu.cn>*.

## ***Visualization***

To better demonstrate the segmentation results, we have provided visualization code that not only visualizes the binary mask output of the segmentation network but also compares the ground truth with our segmentation results. These will be clearly displayed on the skin lesion images, where the green outline represents the ground truth, and the red outline represents the predicted segmentation results from our network.

![Visual](/demo/visual.png)

We have packaged the visualization tools, and you only need to run [`out.py`](/output/out.py) to use them.

## ***References***
1) [**TransUnet**](https://github.com/Beckschen/TransUNet)
2) [**Field of Junctions**](https://github.com/dorverbin/fieldofjunctions)

## ***Citation***

```
```
