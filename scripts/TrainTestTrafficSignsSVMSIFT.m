clc; clear; close all;

img_size = 40;

dataPath = "Data";

tic;
TrainTrafficSignsSVMSIFT(dataPath, img_size);
dt_train = toc;

fprintf("Training Time SVM SIFT: %f\n", dt_train);

%%
img_size = 40;

dataPath = "Data";
tic;
[accuracy, confusionMat, precision, recall, f1Score] = TestTrafficSignsSVMSIFT(dataPath, img_size);
dt_test = toc;

fprintf("Testing Time SVM SIFT: %f\n", dt_test);