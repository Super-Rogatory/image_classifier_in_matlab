% MATH 320 Project Code:
% Chukwudi Ikem | Arsal Khan

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

%% Build the machine learning model.
% Defining variables. 
% filters specifies the number of output features we want
% kernel_size or filter_size determines the output features
% recall: each image is 32x32x3
num_of_channels = 3;
filters = 32;
kernel_size = [3 3];
input_shape = [32 32 3];
pool_size = [2 2];
units = 512;
% Inputs 2-d images to a network and normalizes the dataset in one function.
% This will prove useful when all the layers are applied.
input_layer = imageInputLayer(input_shape);
main_layers = [
    % 2-D convolutional layer that computes dot product of the weights and
    % the input
    convolution2dLayer(kernel_size, filters, 'Padding','same');
    % activation function to produce an output based on previous input
    reluLayer();
    % performs downsampling, computing the maximum value at each "region"
    maxPooling2dLayer(pool_size);
    reluLayer();
    convolution2dLayer(kernel_size, filters, 'Padding','same');
    reluLayer();
    maxPooling2dLayer(pool_size);
    convolution2dLayer(kernel_size, 2 * filters, 'Padding','same');
    reluLayer();
    maxPooling2dLayer(pool_size);
];
final_layers = [
    % fully connected layer multiplies input by a weight matrix and adds
    % bias.
    fullyConnectedLayer(units);
    reluLayer;
    % randomly drops about half of the neurons
    dropoutLayer(0.2);
    fullyConnectedLayer(10);
    % activation function - softmax
    softmaxLayer;
    % computers cross-entropy loss
    classificationLayer;
];
layers = [input_layer; main_layers; final_layers];
layers(2).Weights = 0.0001 * randn([kernel_size num_of_channels filters]);

%% Train the machine learning model. 
training_settings = trainingOptions('sgdm', 'InitialLearnRate', 0.001, 'MaxEpochs', 1, 'MiniBatchSize', 64, 'Verbose', true);

cifar10_compiled = trainNetwork(trainingImages, trainingLabels, layers, training_settings);
%% Test the machine learning model. 
% Run the network on the test set.
test_outcome = classify(cifar10_compiled, testImages);

% Calculate the accuracy.
accuracy = sum(test_outcome == testLabels)/numel(testLabels)

