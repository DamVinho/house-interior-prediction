# house-interior-prediction
This project aims at classifying house interior as modern (M) or old (O).
The dataset comes from Kaggle "House  Rooms Image Dataset" and contains
5250 images from 5 different rooms (Bathroom, Bedroom, Dinning, Kitchen,
and Livingroom). The dataset is not so well balanced as some rooms
contains more images than others, e.g: bathroom contains 606 images
vs 1248 for bedroom.
Also note that the dataset is unlabelled, however, a hand labelled file of
about 450 images is provided by
this (https://github.com/V-Sher/house-interior-prediction) github repo.
Note  that these label are only for the first 450 images in bedrooms.
This is quite small but let see what we can achieve. 

At the first time, I'll explore the data, build a model from scratch
and try to improve the model using image augmentation.

At the second time, I'll use transfer learning and fine-tunning to
further improve the performance.

At the third time, I'll train a model using semi-supervised learning 
to make use of all the data available
