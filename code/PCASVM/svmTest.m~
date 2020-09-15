function [testResults] = svmTest(trainedModel,testBirds,type)
    if(type == "Features") testMatrix = [testBirds.Features];end
    if(type == "Spectograms") testMatrix = [testBirds.Spectogram];end
    testBCIDs = [testBirds.BirdCodeID];
    classes = unique(testBCIDs);
    svmModel = trainedModel;
    testSize = size(testMatrix,2);
    svmTestData = testMatrix';
    testResults = strings([1 testSize]);
    correct = 0;
    for test = 1:testSize 
        svmTestVec = svmTestData(test,:);
        for j = 1:numel(classes) %For each test run it through each classfier
            [~,score] = predict(svmModel{j},svmTestVec);
            Scores(:,j) = score(:,2);
        end
        [~,svmPredBC] = max(Scores,[],2);
        pcaBC = testBCIDs(test);
        if(pcaBC == svmPredBC)
            correct = correct +1;
        end
        predBC = "";
        if(svmPredBC == 1) 
            predBC = "CAQU";end
        if(svmPredBC == 2) 
            predBC = "DEJU";end
        if(svmPredBC == 3) 
            predBC = "OCWA";end
        if(svmPredBC == 4) 
            predBC = "STJA";end
        if(svmPredBC == 5) 
            predBC = "WREN";end
        if(svmPredBC == 6) 
            predBC = "PSFL";end
        testResults(test) = predBC;
    end
end