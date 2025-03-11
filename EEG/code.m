% Load the data from the .mat files
FADHD = load('FADHD.mat');
FC = load('FC.mat');

% Extract the EEG data from the .mat files
FADHD_data = FADHD.FADHD;
FC_data = FC.FC;

% Assuming FC_data is a 1x11 cell array
FC_data_list = cellfun(@(cell) cell{1}, num2cell(FC_data, 1), 'UniformOutput', false);

% Convert cell array to a list of 27x5120 arrays
FADHD_data_list = cellfun(@(cell) cell{1}, num2cell(FADHD_data, 1), 'UniformOutput', false);

% Find the size of the first element in FC_data_list
min_size = size(FC_data_list{1});

% Resize FC_data_list to have consistent dimensions
FC_data_resized = cat(3, cellfun(@(data) imresize3(data, min_size), FC_data_list, 'UniformOutput', false));

% Convert FADHD_data_list to a 4D array
FADHD_data_array = cat(4, FADHD_data_list{:});

% Replace the sample data with the actual EEG data
X = cat(4, FADHD_data_array, FC_data_resized);

% Create labels (1 for FADHD, 0 for FC)
y = [ones(1, numel(FADHD_data_list)), zeros(1, size(FC_data_resized, 3))];

% Split the data into training and testing sets (80:20 ratio)
rng(42); % For reproducibility
idx = randperm(numel(y));
split_idx = round(0.8 * numel(y));

X_train = X(:, :, :, idx(1:split_idx));
y_train = y(:, idx(1:split_idx));

X_test = X(:, :, :, idx(split_idx+1:end));
y_test = y(:, idx(split_idx+1:end));

% Define your CNN architecture
layers = [
    convolution2dLayer(3, 32, 'Activation', 'relu', 'InputSize', [min_size 1])
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 64, 'Activation', 'relu')
    maxPooling2dLayer(2, 'Stride', 2)
    fullyConnectedLayer(64, 'Activation', 'relu')
    fullyConnectedLayer(2, 'Activation', 'softmax')
    classificationLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Create and train the network
net = trainNetwork(X_train, categorical(y_train), layers, options);

% Evaluate the network
y_pred = classify(net, X_test);
accuracy = sum(y_pred == categorical(y_test)) / numel(y_test);
fprintf('Test accuracy: %.2f%%\n', accuracy * 100);
