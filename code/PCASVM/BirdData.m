classdef BirdData
    properties
        ID, 
        Features,
        BirdCode,
        BirdCodeID,
        Spectogram
    end
    methods(Static)
        function id = setBirdCodeID(birdcode)
            if(birdcode == "CAQU")
                id = 1;
            end
            if(birdcode == "DEJU")
                id = 2;
            end
            if(birdcode == "OCWA")
                id = 3;
            end
            if(birdcode == "STJA")
                id = 4;
            end
            if(birdcode == "WREN")
                id = 5;
            end
            if(birdcode == "PSFL")
                id = 6;
            end
        end
end
end