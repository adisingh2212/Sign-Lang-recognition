This folder contains files for the training and testing the I3D model on the ASL Citizen Dataset.  

### File description:

**pytorch_i3d.py**: This is the network file, containing code for i3d architecture. It was sourced from: [link]

**videotransforms.py**: This contains code for videotransforms that are applied while training and testing such as random crop and center crop.

**aslcitizen_dataset.py**: This is the dataset file. It loads the videos from data csvs, downsampling and padding videos as needed, and returns samples for training and testing along with one-hot encoded labels.

**aslcitizen_training.py**: This file contains code to train an I3D model on the ASLCitizen dataset. 

**aslcitizen_testing.py**: This file contains code to test the trained I3D model on the ASLCitizen test dataset. For the task of dictionary retrieval, the goal is to get a ranked list of glosses for any given video. This code outputs the average top-1, top-5, top-15, top-20 accuracy, Discounted Cumulative Gain (DCG), Mean Reciprocal Rank on the whole dataset. It additionally outputs two kinds of confusion matrices to help with analysis -- one complete confusion matrix, and a confusion matrix mini that highlights top five confusions for a given gloss.

### Instructions:

Open aslcitizen_training.py and update the file paths to dataset and datacsvs as needed (lines 35-38). Update names for log and weights folders as needed (lines 40-43). 

To train, use the following command on the command line. With a single GPU, it takes approximately 3 days to complete training. 
```
python3 aslcitizen_training.py 
```
Once done with training, you can find the saved model weights in a folder (*'saved_weights'* as default). The weights are named such that the last set of digits is the validation accuracy (e.g., *'trainingfull_jan1a75_0.736444.pt'* had validation accuracy of 73.64%). 

To evaluate chosen model weights, open aslcitizen_testing.py. Update the file paths to dataset and datacsvs as needed (lines 55-58). Update names for output files as needed (lines 60-61). Update path to model weights (line 83). 

To test, use the following command on the command line. With a single GPU, it takes approximately 45 minutes to complete testing.
```
python3 aslcitizen_testing.py
```
The results can be found in .txt and .csv files generated in the same folder. 

~~To run the subset experiments on the trained model weights, open subset.py. Update the file paths to dataset and datacsvs as needed (lines 54-56). Update names for output files as needed (lines 57-58). Update path to model weights (line 79).~~

~~To test subset, use the following command on the command line:~~
```
~~python3 subset.py~~
```

