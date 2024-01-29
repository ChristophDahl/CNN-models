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

windowProportion = [100, 20, 10, 8, 6, 4, 2, 1];
Output = [];
for i = 1:length(windowProportion)
    
    imds = imageDatastore(personFoldersWithMoreThanXImages, 'IncludeSubfolders', true, 'FileExtensions', '.jpg', 'LabelSource', 'foldernames');
    input_size = [224, 224, 3];  % Change the size as needed
    
    customReadFunction = @(filename) customReadAndResize(filename, input_size,windowProportion(i));
    
    imds.ReadFcn = customReadFunction;
        
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
    Output{i}.windowProportion = windowProportion(i);
end
cd(resultsDirectory)
save('ResultsCoverObjectParts','Output')



function I = customReadAndResize(filename, targetSize, windowProportion)
    I = imread(filename);
    
    % Get the size of the image
    [height, width, ~] = size(I);

    % Calculate the center of the image
    centerX = round(width / 2);
    centerY = round(height / 2);
    
    jitterRange = 10;
    jitterX = randi([-jitterRange, jitterRange], 1);
    jitterY = randi([-jitterRange, jitterRange], 1);
    centerX = centerX + jitterX;
    centerY = centerY + jitterY;
  
    
    halfWindowSizeY = floor(height / windowProportion);
    halfWindowSizeX = floor(width / windowProportion);

    startX = max(1, centerX - halfWindowSizeX/2);
    startY = max(1, centerY - halfWindowSizeY/2);
    endX = min(width, centerX + halfWindowSizeX/2);
    endY = min(height, centerY + halfWindowSizeY/2);
    
%     I = I(startY:endY, startX:endX, :);
%     midGreyValue = 255; % 128 for mid-grey in uint8 format
%     imageWithGreyBackground = uint8(midGreyValue * ones(size(I)));

    % Place the center window on the mid-grey background
    I(startY:endY, startX:endX, :) = 1;
%     I = imageWithGreyBackground;
    
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
    clear imageWithGreyBackground
%     I = imrotate(I, (rotationalAngle),'bilinear','crop');
    %I = rgb2gray(I);
%     I = rgb2gray(I);
%     I = cat(3,I,I,I);
    if ~isa(I, 'uint8')
        I = uint8(I);
    end
end

