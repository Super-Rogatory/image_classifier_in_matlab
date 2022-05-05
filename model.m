% MATH 320 - Building the CNN model
% Chukwudi Ikem
% Arsal Khan

%% Process data from our data set.
% Reads every file you *should* have in cifar-10-batches-mat. And returns
% the train images as well as the train labels
% print working directory into current_directory
current_directory = pwd;
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
% Downloads CIFAR10 dataset from url
helperCIFAR10Data.download(url,current_directory);
[trainingImages,trainingLabels,testImages,testLabels] = helperCIFAR10Data.load(current_directory);
% Each image is of size => 32 32 3 (3 color channels, 32 x 32).
% The training images set has the following dimenstions => 32 x 32 x 3 x 50000
% The training labels are in plain english => 50000 x 1

%% Build the machine learning model. NOTE: NOT FINISHED
% Defining variables. 
% filters specifies the number of output features we want
% kernel_size or filter_size determines the output features
% recall: each image is 32x32x3
filters = 32;
kernel_size = [3 3];
input_shape = [32 32 3];
pool_size = [2 2];
units = 1024;
% Inputs 2-d images to a network and normalizes the dataset in one function.
% This will prove useful when all the layers are applied.
input_layer = imageInputLayer(input_shape);
main_layers = [
    convolution2dLayer(kernel_size, filters, 'Padding','same');
    % activation function to produce an output based on previous input
    reluLayer;
    maxPooling2dLayer(pool_size);
    fullyConnectedLayer(units);
    reluLayer;
    dropoutLayer(0.5);
    fullyConnectedLayer(10);
    softmaxLayer;
    classificationLayer;
];
layers = [input_layer; main_layers];


%% Train the machine learning model. 
training_settings = trainingOptions('adam', 'InitialLearnRate', 0.1, 'MaxEpochs', 10, 'MiniBatchSize', 64, 'Verbose', true);

cifar10_compiled = trainNetwork(trainingImages, trainingLabels, layers, training_settings);
%% Test the machine learning model. 
% Run the network on the test set.
test_outcome = classify(cifar10_compiled, testImages);

% Calculate the accuracy.
accuracy = sum(test_outcome == testLabels)/numel(testLabels)