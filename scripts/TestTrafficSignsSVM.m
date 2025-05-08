function [accuracy, confusionMat, precision, recall, f1Score] = TestTrafficSignsSVM(testPath, img_size)
    % Tests the trained SVM model on new images and evaluates performance
    if nargin < 1 || isempty(testPath)
        testPath = "Data";  
    end
    if nargin < 2 || isempty(img_size)
        img_size = 40;
    end
    basePath = testPath;
    % Load trained SVM model
    if exist('TrafficSignSVM.mat', 'file')
        load('TrafficSignSVM.mat', 'SVMModel');
    else
        error('Trained model not found. Run TrainTrafficSignsSVM first.');
    end

    % Load test images
    correct = 0;

    % Initialize arrays for storing the true labels and predicted labels
    trueLabels = [];
    predictedLabels = [];

    testingFile = readtable(testPath + "\Test.csv"); % The CSV has columns: Width, Height, Roi.X1, Roi.Y1, Roi.X2, 
                                              % Roi.Y2, ClassId, Path
  
    n = size(testingFile, 1);

    for i = 1:n
        imgPath = testingFile.Path(i);
        fileName = split(imgPath{1}, "/");
        fileName = fileName{end};
        
        ImgFile = fullfile(basePath, imgPath); 
        groundTruthLabel = testingFile.ClassId(i);
        RoiX1 = testingFile.Roi_X1(i);
        RoiY1 = testingFile.Roi_Y1(i);
        RoiX2 = testingFile.Roi_X2(i);
        RoiY2 = testingFile.Roi_Y2(i);
        Img = imread(ImgFile);


        % Crop out the region of interest, and convert to grayscale 
        Img = Img(RoiY1 + 1:RoiY2 + 1, RoiX1 + 1:RoiX2 + 1);

        % Resize the image to a fixed size
        Img = imresize(Img, [img_size img_size]);

        % Extract HOG features
        features = extractHOGFeatures((Img));


        % Predict the label using the SVM model
        predictedLabel = predict(SVMModel, features);

        % Store the true and predicted labels
        trueLabels = [trueLabels; groundTruthLabel];
        predictedLabels = [predictedLabels; predictedLabel];

        % Display the prediction result
        if (rem(i, 1000) == 0)
            fprintf('Image: %s | Predicted Class: %d | Ground Truth: %d\n', fileName, predictedLabel, groundTruthLabel);
        end

        
        if predictedLabel == groundTruthLabel
            correct = correct + 1;
        end
    end

    % Convert predictedLabels to double to match trueLabels
    predictedLabels = double(predictedLabels);

    % Calculate evaluation metrics
    accuracy = correct / n;
    fprintf('Accuracy: %.2f%%\n', accuracy * 100);

    % Compute confusion matrix
    confusionMat = confusionmat(trueLabels, predictedLabels);
    fprintf('Confusion Matrix:\n');
    disp(confusionMat);
    % Plot heatmap
    figure;
    imagesc(confusionMat); % Display matrix as an image
    colormap('hot'); % Choose a colormap ('hot', 'jet', 'parula', etc.)
    colorbar; % Add a colorbar for reference

% Label axes
xlabel('Predicted Labels');
ylabel('True Labels');
title('Confusion Matrix Heatmap');

    % Precision, Recall, F1-Score Calculation
    precision = diag(confusionMat) ./ sum(confusionMat, 2);
    recall = diag(confusionMat) ./ sum(confusionMat, 1)';
    recall(isnan(recall)) = 0;
    f1Score = 2 * (precision .* recall) ./ (precision + recall);
    
    % Display precision, recall, and F1-score for each class
    for i = 1:length(precision)
        fprintf('Class %d - Precision: %.2f | Recall: %.2f | F1-Score: %.2f\n', i-1, precision(i), recall(i), f1Score(i));
    end

    % Calculate macro-averaged precision, recall, and F1-score
    macroPrecision = mean(precision);
    macroRecall = mean(recall);
    macroF1Score = mean(f1Score);

    % Display macro-averaged metrics
    fprintf('Macro-averaged Precision: %.2f\n', macroPrecision);
    fprintf('Macro-averaged Recall: %.2f\n', macroRecall);
    fprintf('Macro-averaged F1-Score: %.2f\n', macroF1Score);

    fprintf('Testing completed.\n');
end
