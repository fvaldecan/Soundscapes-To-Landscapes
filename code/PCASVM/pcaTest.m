function [testResults] = pcaTest(trainBirds,eigenCoeffs,eigenVectors,meanA, testBirds,type)
    if(type == "Features") testMatrix = [testBirds.Features]; end
    if(type == "Spectograms") testMatrix = [testBirds.Spectogram]; end
    testSize = size(testMatrix,2);
    eigcoeffs_training = eigenCoeffs;
    N = size(eigcoeffs_training,1);
    %Get best predicted set
    testResults = strings([1 testSize]);
    bestRate = 0;
    for k = [round(N/4) round(N/2) N] %Find best K
        result = "";
        rec_rate = 0;
        for j = 1:testSize 
            featVec = testMatrix(:,j);   
            testBC = testBirds(j).BirdCode;
            eigcoeffs_im = eigenVectors*(featVec -meanA);
            coeff = repmat(eigcoeffs_im,1,N);
            diffs = eigcoeffs_training - coeff;
            diffs = sum(diffs(1:k,:).^2,1);
            [~,minindex] = min(diffs);
            predBC = trainBirds(minindex).BirdCode;
            if(testBC == predBC) rec_rate = rec_rate + 1; end
            result(j) = predBC; 
        end
        rec_rate = (rec_rate/testSize)*100;
        if(rec_rate > bestRate)
            bestRate = rec_rate;
            testResults = result;
        end
    end
end