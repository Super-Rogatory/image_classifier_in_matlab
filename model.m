% MATH 320 - Building the CNN model
% Chukwudi Ikem
% Arsal Khan

% Reads every file you *should* have in cifar-10-batches-mat. And returns
% the train images as well as the train labels

cifar10Data = append(pwd);
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';

helperCIFAR10Data.download(url,cifar10Data);
[trainingImages,trainingLabels,testImages,testLabels] = helperCIFAR10Data.load(cifar10Data);
% 32 32 3 50000 size
trainingImages(1, 1, : ,1)
