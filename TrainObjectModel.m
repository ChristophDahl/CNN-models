clear all
clc

resultsDirectory = 'D:\studies\computationalVision\viewDependencyFaceObject\results';
datasetDirectory = 'D:\studies\computationalVision\viewDependencyFaceObject\databases\objectDatabase\caltech-101\101_ObjectCategories';
% dataStorage = 'R:\science\computationalVision\ClassifyingWildFaces\results';
personFolders = dir(datasetDirectory);
personFolders = personFolders([personFolders.isdir]);  % Only keep directories
personFoldersWithMoreThanXImages = {};

for i = 1: 60%numel(personFolders)
    personFolder = personFolders(i).name;
    personFolderPath = fullfile(datasetDirectory, personFolder);
    imageFiles = dir(fullfile(personFolderPath, '*.jpg'));  % Adjust the extension if needed
    if numel(imageFiles) > 10
        personFoldersWithMoreThanXImages = [personFoldersWithMoreThanXImages, personFolderPath];
    end
end

imds = imageDatastore(personFoldersWithMoreThanXImages, 'IncludeSubfolders', true, 'FileExtensions', '.jpg', 'LabelSource', 'foldernames');
input_size = [224, 224, 3];  % Change the size as needed

customReadFunction = @(filename) customReadAndResize(filename, input_size);

imds.ReadFcn = customReadFunction;

splitRatio = 0.8;
[imdsTrain, imdsTest] = splitEachLabel(imds, splitRatio, 'randomized');

% % Data augmentation parameters
% augmentation = imageDataAugmenter( ...
%     'RandRotation',[-20,20], ...
%     'RandXReflection',true, ...
%     'RandYReflection',true, ...
%     'RandXScale',[0.8,1.2], ...
%     'RandYScale',[0.8,1.2]);

% % Create augmented image datastore for training set
% augImdsTrain = augmentedImageDatastore(input_size(1:2), imdsTrain, 'DataAugmentation', augmentation);

num_classes = numel(categories(imdsTrain.Labels));
% num_epochs = 80;
% batch_size = 64; 
% initial_learn_rate = 0.001; 
% validation_freq = 10; 

image_height = input_size(1);
image_width = input_size(2);
num_channels = input_size(3);   

layers = [
    imageInputLayer([image_height image_width num_channels])

convolution2dLayer(3, 64, 'Padding', 'same')
batchNormalizationLayer
reluLayer()
convolution2dLayer(3, 64, 'Padding', 'same')
batchNormalizationLayer
reluLayer()    
maxPooling2dLayer(2, 'Stride', 2)

convolution2dLayer(3, 128, 'Padding', 'same')
batchNormalizationLayer
reluLayer()
convolution2dLayer(3, 128, 'Padding', 'same')
batchNormalizationLayer
reluLayer()
maxPooling2dLayer(2, 'Stride', 2)

convolution2dLayer(3, 256, 'Padding', 'same')
batchNormalizationLayer
reluLayer()
convolution2dLayer(3, 256, 'Padding', 'same')
batchNormalizationLayer
reluLayer()
convolution2dLayer(3, 256, 'Padding', 'same')
batchNormalizationLayer
reluLayer()
maxPooling2dLayer(2, 'Stride', 2)

convolution2dLayer(3, 512, 'Padding', 'same')
batchNormalizationLayer
reluLayer()
convolution2dLayer(3, 512, 'Padding', 'same')
batchNormalizationLayer
reluLayer()
convolution2dLayer(3, 512, 'Padding', 'same')
batchNormalizationLayer
reluLayer()
maxPooling2dLayer(2, 'Stride', 2)

convolution2dLayer(3, 512, 'Padding', 'same')
batchNormalizationLayer
reluLayer()
convolution2dLayer(3, 512, 'Padding', 'same')
batchNormalizationLayer
reluLayer()
convolution2dLayer(3, 512, 'Padding', 'same')
batchNormalizationLayer
reluLayer()
maxPooling2dLayer(2, 'Stride', 2)

fullyConnectedLayer(4096)
batchNormalizationLayer
reluLayer()
dropoutLayer(0.5)

fullyConnectedLayer(4096)
batchNormalizationLayer
reluLayer()
dropoutLayer(0.5)

fullyConnectedLayer(4096)
batchNormalizationLayer
reluLayer()
dropoutLayer(0.5)

fullyConnectedLayer(num_classes) % Assuming you have defined numClasses
softmaxLayer()
classificationLayer
];
% options = trainingOptions('adam', ...
%     'MaxEpochs', num_epochs, ...
%     'MiniBatchSize', batch_size, ...
%     'InitialLearnRate', initial_learn_rate, ...
%     'LearnRateSchedule', 'piecewise', ...
%     'Shuffle', 'every-epoch', ...
%     'Plots', 'training-progress', ...
%     'ValidationData', imdsTest, ...
%     'ValidationFrequency', validation_freq, ...
%     'Verbose', true, ...
%     'Plot', 'none');

options = trainingOptions('sgdm', ...
    'MaxEpochs', 80, ...
    'MiniBatchSize', 16, ...
    'InitialLearnRate', 0.001, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'none', ...
    'ValidationData', imdsTest, ...
    'ValidationFrequency', 10, ...
    'Verbose', true);

net = trainNetwork(imdsTrain,layers,options);

cd(resultsDirectory)
%save the network
save('WildObjectNetwork', 'net');

YPred = classify(net,imdsTest);
YValidation = imdsTest.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

figure('Position',[200 300 1200 600])
img = readimage(imds,2);
act = activations(net, img, 3);
for u = 1:8
    subplot(5,8,u),imshow(act(:,:,u))
end
act = activations(net, img, 7);
for u = 1:8
    subplot(5,8,u+8),imshow(act(:,:,u+8))
end
act = activations(net, img, 11);
for u = 1:8
    subplot(5,8,u+16),imshow(act(:,:,u+16))
end
act = activations(net, img, 15);
for u = 1:8
    subplot(5,8,u+24),imshow(act(:,:,u+24))
end
act = activations(net, img, 19);
for u = 1:8
    subplot(5,8,u+32),imshow(act(:,:,u+32))
end

function I = customReadAndResize(filename, targetSize)
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