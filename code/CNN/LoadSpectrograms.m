

function LoadSpectrograms(pad, imagePath)
%% Setup the Import Options and import the data

opts = delimitedTextImportOptions("NumVariables", 20);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["filename", "recording", "site", "device", "year"...
    , "month", "day", "hour", "min", "species", "birdcode", "commonname"...
    , "songtype", "x1", "x2", "y1", "y2", "score", "vote", "type"];
opts.VariableTypes = ["string", "string", "string", "string", "double"...
    , "double", "double", "double", "double", "string", "string", "string"...
    , "string", "double", "double", "double", "double", "double", "string", "string"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["filename", "recording"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["filename", "recording", "site", "device", "species", "birdcode"...
    , "commonname", "songtype", "vote", "type"], "EmptyFieldRule", "auto");
%%
% Import the data
T = readtable("FinalProjectBagOfWords/common/trunc_pattern_matching_ROIs_200331.csv", opts);
m = table2array(T);
x = str2double(m(:,15))-str2double(m(:,14));
y = str2double(m(:,17))-str2double(m(:,16));



%Image Files
folder1 = 'FilesPt1';
filesPt1 = dir(fullfile(folder1,'*.wav'));

folder3 = 'FilesPt3';
filesPt3 = dir(fullfile(folder3,'*.wav'));

if pad == 0
    for j = 1:2000
        %Read from folder containing wavfiles
        baseFileName = filesPt3(j).name;
        baseFileNameSize = size(baseFileName);
        csvFileValue = baseFileName(1:baseFileNameSize(2)-4);
        birdCodeIndex = find(strcmp(m(:,1), baseFileName));
        birdCode = m(birdCodeIndex(1),11);

        %Get Time Stamps for region of interest in wavfile
        ROIx1 = str2double(m(birdCodeIndex(1),14));
        ROIx2 = str2double(m(birdCodeIndex(1),15));

        %Create folder for bird code if doesnt exist
        if ~exist("FinalProjectBagOfWords/images/FinalProjectSpectrograms/" + birdCode, "dir")
            mkdir("FinalProjectBagOfWords/images/FinalProjectSpectrograms/" + birdCode);
        end

        fullFileName = fullfile(folder3, baseFileName);
        fprintf(1, 'Now reading %s\n', fullFileName);

        [fullData, fs] = audioread(fullFileName); 

        %Cut ROI from wav data
        ROI = fullData(round(ROIx1*fs):round(ROIx2*fs));

        %Form Spectrogran of ROI
        [S,T,F,P] = spectrogram(ROI,256,250,256,fs,'yaxis');
        tF = F.';
        h = pcolor(tF,T,abs(S));
        set(gca, 'Visible', 'off');
        set(h,'EdgeColor','none');

        %Create folder of ROI if doesnt exit
        if ~exist("FinalProjectBagOfWords/images/FinalProjectSpectrograms/"+ birdCode + "_ROIS/", "dir")
                   mkdir("FinalProjectBagOfWords/images/FinalProjectSpectrograms/"+ birdCode + "_ROIS/");
        end
        saveas(h, ("FinalProjectBagOfWords/images/FinalProjectSpectrograms/" + birdCode + "_ROIS/" + csvFileValue + ".jpg"));
    end

else
    %p = length(files);
    for i = 1200:2000
        baseFileName = filesPt1(i).name;
        baseFileNameSize = size(baseFileName);
        csvFileValue = baseFileName(1:baseFileNameSize(2)-4);
        birdCodeIndex = find(strcmp(m(:,1), baseFileName));
        birdCode = m(birdCodeIndex(1),11);

        ROIx1 = str2double(m(birdCodeIndex(1),14));
        ROIx2 = str2double(m(birdCodeIndex(1),15));


        fullFileName = fullfile(folder1, baseFileName);
        fprintf(1, 'Now reading %s\n', fullFileName);

        %%data is sampled data, fs is sample rate
        [fullData, fs] = audioread(fullFileName);    
        ROI = fullData(round(ROIx1*fs):round(ROIx2*fs));

        %[data, fs] = audioread(fullFileName, ROI); 

        %Zero Padding 
        fixedSeconds = 144000;

        sigma= mean(abs(ROI));
        mu = 0;
        s = sigma*randn(fixedSeconds - length(y),1)+mu;
        mid = randi([1 length(s)]);
        padROI = cat(1, s(1 : mid),ROI,s(mid : length(s)));

        [S,T,F,P] = spectrogram(padROI,256,250,256,fs,'yaxis');
        tF = F.';
        h = pcolor(tF,T,abs(S));
        set(gca, 'Visible', 'off');
        set(h,'EdgeColor','none');


    %    spectrogram(data, hamming(512),256,1024,fs,'yaxis');
      %  set(gca, 'Visible', 'off');
        if ~exist("FinalProjectBagOfWords/images/FinalProjectSpectrograms/"+ birdCode + "ROIS_PAD/", "dir")
                   mkdir("FinalProjectBagOfWords/images/FinalProjectSpectrograms/"+ birdCode + "ROIS_PAD/");
        end
        saveas(h, ("FinalProjectBagOfWords/images/FinalProjectSpectrograms/" + birdCode + "ROIS_PAD/" + csvFileValue + ".jpg"));
    end
end

