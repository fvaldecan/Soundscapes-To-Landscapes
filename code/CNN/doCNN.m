function doCNN(net,load, zeropad, transfer,svm)

%%%Function To Run CNN analysis 
%%%Function assumes Wav Files and CSV exist
%%%load,transfer,svm = bool variables

%%% generate ROIS
if load == 1
    if zeropad == 1
        LoadSpectrograms(1);
    else
        LoadSpectrograms(2);
    end
end


%%% NeuralNetwork Transfer Learning Method
if transfer == 1
    cnnTransferLearning(net);
end

%%% Neural Network SVM Method
if svm == 1
    cnnSVM(net);
end

