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
patchSizes = floor([input_size(1)./linspace(1,10,10)]);
Output = [];
for i = 1:length(patchSizes)
    
    imds = imageDatastore(personFoldersWithMoreThanXImages, 'IncludeSubfolders', true, 'FileExtensions', '.png', 'LabelSource', 'foldernames');
    input_size = [224, 224, 3];  % Change the size as needed

    if i == 1
        customReadFunction = @(filename) customReadAndResize2(filename, input_size);
    else
        customReadFunction = @(filename) customReadAndResize(filename, input_size,patchSizes(i));
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
    Output{i}.patchSize = patchSizes(i);
end
cd(resultsDirectory)
save('ResultsScrambledBWFaces','Output')


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

function I = customReadAndResize(filename, targetSize, patchSize)
    I = imread(filename);

    I = imresize(I, [targetSize(1), targetSize(2)]);
    %I = imgaussfilt(I, blurr);
    I = rgb2gray(I);
    I = cat(3,I,I,I);    
    
    I = I(1:floor(size(I, 1)/patchSize)*patchSize, 1:floor(size(I, 2)/patchSize)*patchSize, :);

    % Calculate the number of patches
    numPatchesX = size(I, 2) / patchSize;
    numPatchesY = size(I, 1) / patchSize;

    % Create a cell array of patches
    patches = mat2cell(I, repmat(patchSize,numPatchesY,1), repmat(patchSize,numPatchesX,1), size(I,3));

    % Shuffle the patches
    shuffledIndices = randperm(numel(patches));
    shuffledPatches = patches(shuffledIndices);

    % Reconstruct the shuffled image from the patches
    I = cell2mat(reshape(shuffledPatches, size(patches)));
    I = imresize(I, [targetSize(1), targetSize(2)]);

    if ~isa(I, 'uint8')
        I = uint8(I);
    end
end


function I = customReadAndResize2(filename, targetSize)
    I = imread(filename);
    I = rgb2gray(I);
    I = cat(3,I,I,I);
    
    I = imresize(I, [targetSize(1), targetSize(2)]);
    if ~isa(I, 'uint8')
        I = uint8(I);
    end
end

