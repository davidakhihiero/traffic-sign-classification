clc; clear; close all;

img_size = 40;

dataPath = "Data";

tic;
TrainTrafficSignsCNN(dataPath, img_size);
dt_train = toc;

fprintf("Training Time CNN: %f\n", dt_train);

%%
tic;
[accuracy, confusionMat, precision, recall, f1Score] = TestTrafficSignsCNN(dataPath, img_size);
dt_test = toc;

fprintf("Testing Time CNN: %f\n", dt_test);