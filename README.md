# Change Detection
 
<p align="left"> 
We have two images one in the past and new one for the same region and we need to detect the difference between them if new building added or some of them removed  
</p>

## 📝 Table of Contents

- [About](#about)
- [Preprocessing](#preprocessing)
- [Model](#Model)
- [Training](#training)
- [Testing and Accuracy](#testing-and-accuracy)

## About <a name = "about"></a>
 - we try many implementations for this problem many approaches classical and deep learning .
### Classica 
 - we try to make erosion and diolation to make mask to detect the buildings each image then make images difference or rational
 - detect the edges to detect the building
 - Apply thresholding in each image to detect the building
 - Apply CVA
### Deep learning
- we apply GANS approach generator and discriminator

## Preprocessing <a name = "preprocessing"></a>
  - Read the two images and Resize them to 512 * 512 
  - Concatenate them together to get vector with 6 channel
  -  
## Model <a name = "Model"></a>
  - Generator we use Unet for 5 layer until the feature size 1024 the use convTranspose to make the image 512*512
  - Discriminator we make many convelution layes 

## Training <a name = "training"></a>
  - Tain the generator 2 times for each batch and the discriminator 2 times on the same batch
  - the trainning process take many epoch in different times as we save and load the model each time

## Testing and Accuracy <a name = "testing-and-accuracy"></a>
  - we use Jaccard metric to detect the result
  - we get 86% score on benchmark test-set
  
