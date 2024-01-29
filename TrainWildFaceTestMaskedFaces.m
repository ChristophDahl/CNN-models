clear all
clc

resultsDirectory = 'D:\kai\faceCNN\facesWithMasks\results';
resultsDirectory2 = 'D:\kai\faceCNN\wildFaces\results';

% if you create subfolders and asign images to these folders, use this
% code. Don't run this again, once the folders have been established.

% cd('D:\kai\faceCNN\facesWithMasks\database\sortedDatabase')
% for i = 1:length(categoryNames)
%     mkdir(string(categoryNames(i)))
% end
% 
% cd(datasetDirectoryTest)
% dx = dir(datasetDirectoryTest);
% directoryLocation = 'D:\kai\faceCNN\facesWithMasks\database\sortedDatabase';
% for i = 1:length(categoryNames)
%     currentCategory = categoryNames(i);
%     for j = 3: length(dx)
%         currentFile = dx(j).name;
%         if strfind(string(currentFile),string(currentCategory)) == 1
%             cd()
%             fileToBeCopied = string(currentFile);
%             copyfile(strcat(datasetDirectoryTest,'\',fileToBeCopied),strcat(directoryLocation,'\',string(currentCategory),'\',fileToBeCopied));
%             cd(datasetDirectoryTest)
%         end
%     end
% end

% datasetDirectory = 'D:\kai\faceCNN\wildFaces\database\lfw-deepfunneled';
datasetDirectoryTest = 'D:\kai\faceCNN\facesWithMasks\database\sortedDatabase';
% dataStorage = 'R:\science\computationalVision\ClassifyingWildFaces\results';
personFolders = dir(datasetDirectoryTest);
personFolders = personFolders([personFolders.isdir]);  % Only keep directories
personFoldersWithMoreThanXImages = {};

for i = 1:numel(personFolders)
    personFolder = personFolders(i).name;
    personFolderPath = fullfile(datasetDirectoryTest, personFolder);
    imageFiles = dir(fullfile(personFolderPath, '*.jpg'));  % Adjust the extension if needed
    if numel(imageFiles) >= 1
        personFoldersWithMoreThanXImages = [personFoldersWithMoreThanXImages, personFolderPath];
    end
end

imds = imageDatastore(personFoldersWithMoreThanXImages, 'IncludeSubfolders', true, 'FileExtensions', '.jpg', 'LabelSource', 'foldernames');
categoryNames = unique(imds.Labels);

input_size = [224, 224, 3];  % Change the size as needed

customReadFunction = @(filename) customReadAndResize(filename, input_size);

imds.ReadFcn = customReadFunction;

% splitRatio = 0;
% [imdsTrain, imdsTest] = splitEachLabel(imds, splitRatio, 'randomized');

% do we need to shuffle the presentation order?

image_height = input_size(1);
image_width = input_size(2);
num_channels = input_size(3);   

cd(resultsDirectory2)
load('WildFaceNetwork.mat')

% evaluate the wildFaceNetwork with masked faces.
Output = [];
Output.YPred = classify(net,imds);
Output.YValidation = imds.Labels;
Output.accuracy = sum(Output.YPred == Output.YValidation)/numel(Output.YValidation);
cd(resultsDirectory)
save('ResultsFacesWithMasks','Output')


function I = customReadAndResize(filename, targetSize)
    I = imread(filename);

I = imresize(I, [targetSize(1), targetSize(2)]);
%I = rgb2gray(I);

if ~isa(I, 'uint8')
    I = uint8(I);
end
end

