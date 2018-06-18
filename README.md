Link to Research Article: https://arxiv.org/pdf/1705.02304.pdf 

Title: Deep Speaker: an End-to-End Neural Speaker Embedding System 

Framework: Keras with Tensorflow as Backend

Dataset used in paper: UIDs, XiaoDu, MTurk (None available online)
Dataset used in my implementation: LibriSpeech

Link to Dataset: http://www.openslr.org/12/ (I have not included the dataset since it is >10GB) 
Trained the model with 251 speakers

Lowest Loss reported for Convolutional Model: 1.23
Lowest Loss reported for Recurrent Model: 1.34
Loss for Softmax Pretraining + triplet loss: Not yet combined 


List of Files:

input.py - Includes reading an audio file, preprocessing and loading batches of triplets for inputs to model

conv_model.py - Convolutional Resnet network Implementation

recurrent_model.py - GRU network Implementation

Pretraining.py - Softmax Pretraining for the models triplet_loss.py - Implementation of triplet loss for the network


Corrections to be done/Modification to be added:

1) Batch-size: Although triplet loss tends to converge with larger batches, due to memory constraints a batch size of 12 (12*3 for triplets) was used.

2) Generating Triplets: Generates random triplets from the dataset. Modification needed to generate hard and semi- hard triplets.

3) Triplet Loss: Standard Triplet loss implementation. Modification needed to make it lossless.

4) Parameters for audio processing: Needs work to extract the best behaviour of the model.

5) Data Set: trained the model on a subset of training data. The complete set of data could be
used to get better convergence

6) Truncation: Truncated audio signal to 3 seconds due to memory constraints. Can include the
whole audio to get better results.

7) Generating Pairs: Generating all pairs will make the problem trivial , since the set will contain a
lot of easy pairs when compared to hard pairs. Can modify the network to work on pairs for
comparison.

8) Convolutional model: The inputs to the model can be varied in size to compare results. Used the minimum size possible.

9) Imbalance in Classes: Dataset needs to be accounted for the imbalance in classes.
Oversampling, under sampling or stratified sampling could be adopted.
