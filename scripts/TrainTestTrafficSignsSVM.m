clc; clear; close all;

img_size = 40;

dataPath = "Data";

tic;
TrainTrafficSignsSVM(dataPath, img_size);
dt_train = toc;

fprintf("Training Time SVM: %f\n", dt_train);

%%
img_size = 40;

dataPath = "Data";
tic;
[accuracy, confusionMat, precision, recall, f1Score] = TestTrafficSignsSVM(dataPath, img_size);
dt_test = toc;

fprintf("Testing Time SVM: %f\n", dt_test);