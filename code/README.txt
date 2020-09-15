PCASVM
"In the main.mlx file it should be split into multiple sections. 
Run in the live script the first section to grab the data from a .csv to turn it into a table.
In the next section the one line command references the getBirdData function in the getBirdData.m file. 
This function gets data to be used in the training and testing functions and stores it into BirdData objects.
The next two sections will be used to call for training then testing. svmTrain/svmTest and pcaTrain/pcaTest is provided. 
Lastly the final section outputs the confusion matrix, avgf1Score, and f1scores of the results.

CNN
doCNN.m
Function file that takes five inputs
The inputs are:
net - string value, choose which neural net to use, choices ->"alexnet", "resnet18","googlenet","custom"
load - bool value, set true if you want to load spectrograms, (images and path already set in folder so no need)
zeropad - bool value, set true if you want to load zeropadspectrograms in load function
transfer - bool value, set true if you want to run transfer learning with net given
svm - bool value, set true if you want to run feature extraction to svm with net given

example:
doCNN("resnet18",0,0,1,0)

*above command will run transfer learning using resnet18



The .mlx files are safety in case the m files aren't ready as an easy function for the user to use

LoadSpectrograms.mlx
LoadSpectrograms is split into two for loops, one that creates ROI spectrograms, unpadded, and one that
creates padded Spectrograms. This file assumes you have given it the path to the where thee wav files that
it is forming spectrograms of, are. 

cnnSVM.mlx
This file performs feature extraction on a cnn. There are four different Neural Networks that can be tried 
out and you will have to uncomment out which ones you want to use

cnnTransferLearning.mlx
This file performs Transfer Learning through a CNN. There are four different Neural Networks that can be tried 
out and you will have to uncomment out which ones you want to use

Spectrograms
Spectrograms are held in path ../images/FinalProjectSpectrograms