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

input_size = [224, 224, 3];  % Change the size as needed
windowProportion = [100, 20, 10, 8, 6, 4, 2, 1];
Output = [];
for i = 1:length(windowProportion)
    
    imds = imageDatastore(personFoldersWithMoreThanXImages, 'IncludeSubfolders', true, 'FileExtensions', '.png', 'LabelSource', 'foldernames');
    input_size = [224, 224, 3];  % Change the size as needed

%     if i == 1
%         customReadFunction = @(filename) customReadAndResize2(filename, input_size);
%     else
        customReadFunction = @(filename) customReadAndResize(filename, input_size,windowProportion(i));
%     end
    
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
save('ResultsCoverFaceParts','Output')

function I = customReadAndResize(filename, targetSize, windowProportion)
    I = imread(filename);
    
    % Get the size of the image
    [height, width, ~] = size(I);

    % Calculate the center of the image
    centerX = round(width / 2);
    centerY = round(height / 2);
    
    jitterRange = 20;
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
    I = imresize(I, [targetSize(1), targetSize(2)]);
    if ~isa(I, 'uint8')
        I = uint8(I);
    end
end


% function I = customReadAndResize2(filename, targetSize)
%     I = imread(filename);
% 
%     I = imresize(I, [targetSize(1), targetSize(2)]);
%     if ~isa(I, 'uint8')
%         I = uint8(I);
%     end
% end

