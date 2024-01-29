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

blurringIntensities = [linspace(0,10,11)];
Output = [];
for i = 1:length(blurringIntensities)
    
    imds = imageDatastore(personFoldersWithMoreThanXImages, 'IncludeSubfolders', true, 'FileExtensions', '.jpg', 'LabelSource', 'foldernames');
    input_size = [224, 224, 3];  % Change the size as needed
    
    if i == 1
        customReadFunction = @(filename) customReadAndResize2(filename, input_size);
    else
        customReadFunction = @(filename) customReadAndResize(filename, input_size,blurringIntensities(i));
    end
    
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
    Output{i}.blurringIntensity = blurringIntensities(i);
end
cd(resultsDirectory)
save('ResultsBlurredObjects','Output')


% th = linspace(0, 2*pi, length(vectorAng));                                          % Create Angles
% figure
% polarplot(th, log10(vectorAcc.*100), 'k.-')
% Ax = gca;
% Ax.RTick = (0:.5:numel(th)-1);
% Ax.RTickLabel = compose('10^{%2d}',(2:2:numel(th)-1)-10);
% % Ax.RTickLabel = sprintfc('10^{%2d}',(2:2:numel(th)-1)-10);           
% Ax.ThetaZeroLocation = 'top'
% Ax.ThetaDir = 'clockwise';
% % Ax.RAxis.Label.String = 'Accuracy';
% % subtitle('Rotational angle [degree]','Position',[90,3.8],'HorizontalAlignment','center')
% text(-0.25,0.7,'Accuracy','rotation',90)


function I = customReadAndResize(filename, targetSize, blurr)
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
    I = imresize(I,[targetSize(1), targetSize(2)]);
    if numel(size(I)) == 2
        I = cat(3,I,I,I);
    end
    
    I = imgaussfilt(I, blurr);
    
    clear canvas
    clear resizedImage
    
    %I = rgb2gray(I);

    if ~isa(I, 'uint8')
        I = uint8(I);
    end
end


function I = customReadAndResize2(filename, targetSize)
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
    I = imresize(I,[targetSize(1), targetSize(2)]);
    if numel(size(I)) == 2
        I = cat(3,I,I,I);
    end
    clear canvas
    clear resizedImage
    
    %I = rgb2gray(I);

    if ~isa(I, 'uint8')
        I = uint8(I);
    end
end

