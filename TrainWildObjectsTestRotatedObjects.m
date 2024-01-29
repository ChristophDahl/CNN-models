clear all
clc

resultsDirectory = 'D:\studies\computationalVision\viewDependencyFaceObject\results';
datasetDirectory = 'D:\studies\computationalVision\viewDependencyFaceObject\databases\objectDatabase\caltech-101\101_ObjectCategories';

cd(resultsDirectory)
load('WildObjectNetwork.mat')

personFolders = dir(datasetDirectory);
personFolders = personFolders([personFolders.isdir]);  % Only keep directories
personFoldersWithMoreThanXImages = {};

for i = 1:60 %numel(personFolders)
    personFolder = personFolders(i).name;
    personFolderPath = fullfile(datasetDirectory, personFolder);
    imageFiles = dir(fullfile(personFolderPath, '*.jpg'));  % Adjust the extension if needed
    if numel(imageFiles) >= 1
        personFoldersWithMoreThanXImages = [personFoldersWithMoreThanXImages, personFolderPath];
    end
end

rotationalSteps = [linspace(0,360,25)];
Output = [];
for i = 1:length(rotationalSteps)
    
    imds = imageDatastore(personFoldersWithMoreThanXImages, 'IncludeSubfolders', true, 'FileExtensions', '.jpg', 'LabelSource', 'foldernames');
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
save('ResultsRotatedObjects','Output')



function I = customReadAndResize(filename, targetSize, rotationalAngle)
    I = imread(filename);
    canvasSize = [targetSize(2), targetSize(1)];

    scaleFactorWidth = targetSize(1) / size(I, 2);
    scaleFactorHeight = targetSize(2) / size(I, 1);
    scaleFactor = min(scaleFactorWidth, scaleFactorHeight);
    resizedImage = imresize(I, scaleFactor);

    canvas = uint8(255 * ones(canvasSize(2), canvasSize(1), size(resizedImage, 3)));

    xOffset = max(1, floor((canvasSize(1) - size(resizedImage, 2)) / 2));
    yOffset = max(1, floor((canvasSize(2) - size(resizedImage, 1)) / 2));

    % Calculate the valid region for placing the resized image
    x1 = xOffset;
    y1 = yOffset;
    x2 = xOffset + size(resizedImage, 2) - 1;
    y2 = yOffset + size(resizedImage, 1) - 1;

    % Embed the resized image within the canvas in the valid region
    canvas(y1:y2, x1:x2, :) = resizedImage;
    I = canvas;
    I = imresize(I, [targetSize(1), targetSize(2)]);
    if numel(size(I)) == 2
        I = cat(3,I,I,I);
    end
    clear canvas
    clear resizedImage
    I = imrotate(I, (rotationalAngle),'bilinear','crop');
    %I = rgb2gray(I);

    if ~isa(I, 'uint8')
        I = uint8(I);
    end
end

