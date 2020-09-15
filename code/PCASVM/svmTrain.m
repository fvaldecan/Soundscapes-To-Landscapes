function [svmModel] = svmTrain(trainBirds,type)

    trainBCIDs = [trainBirds.BirdCodeID];
    if(type == "Features")
        trainMatrix = [trainBirds.Features];
    end
    if(type == "Spectograms")
        trainMatrix = [trainBirds.Spectogram];

    end
    svmModel = cell(1,6);
    svmTrainData = trainMatrix';

    rng(1);
    Y = trainBCIDs;
    classes = unique(Y);
    for j = 1:numel(classes)
        % Create classifier for each bird 
        index = (Y == classes(j));
        svmModel{j} = fitcsvm(svmTrainData,index,"ClassNames",[false true],...
            "Standardize",true);
    end
end