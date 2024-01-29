clear all
clc

resultsDirectory = 'D:\studies\computationalVision\viewDependencyFaceObject\results';
datasetDirectory = 'D:\studies\computationalVision\viewDependencyFaceObject\databases\digiFace1M\allfaces';

cd(resultsDirectory)
load('WildFaceNetwork.mat')

personFolders = dir(datasetDirectory);
personFolders = personFolders([personFolders.isdir]);  % Only keep directories
personFoldersWithMoreThanXImages = {};

for i = 1:57 %numel(personFolders)
    personFolder = personFolders(i).name;
    personFolderPath = fullfile(datasetDirectory, personFolder);
    imageFiles = dir(fullfile(personFolderPath, '*.png'));  % Adjust the extension if needed
    if numel(imageFiles) >= 1
        personFoldersWithMoreThanXImages = [personFoldersWithMoreThanXImages, personFolderPath];
    end
end

rotationalSteps = [linspace(0,360,25)];
Output = [];
for i = 1:length(rotationalSteps)
    
    imds = imageDatastore(personFoldersWithMoreThanXImages, 'IncludeSubfolders', true, 'FileExtensions', '.png', 'LabelSource', 'foldernames');
    input_size = [224, 224, 3];  % Change the size as needed
    
    customReadFunction = @(filename) customReadAndResize(filename, input_size,rotationalSteps(i));
    
    imds.ReadFcn = customReadFunction;
    
    % splitRatio = 0.;
    % [imdsTrain, imdsTest] = splitEachLabel(imds, splitRatio, 'randomized');
    
    % do we need to shuffle the presentation order?
    
    image_height = input_size(1);
    image_width = input_size(2);
    num_channels = input_size(3);   
    
    % evaluate the wildFaceNetwork with masked faces.
    YPred = classify(net,imds);
    YValidation = imds.Labels;
    accuracy = sum(YPred == YValidation)/numel(YValidation)
    Output{i}.YPred = YPred;
    Output{i}.YValidation = YValidation;
    Output{i}.accuracy = accuracy;
    Output{i}.rotationalAngle = rotationalSteps(i);
end
cd(resultsDirectory)
save('ResultsRotatedFaces','Output')



function I = customReadAndResize(filename, targetSize, rotationalAngle)
    I = imread(filename);

    I = imresize(I, [targetSize(1), targetSize(2)]);
    I = imrotate(I, (rotationalAngle),'bilinear','crop');
    %I = rgb2gray(I);

    if ~isa(I, 'uint8')
        I = uint8(I);
    end
end

