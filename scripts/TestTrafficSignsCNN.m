function [accuracy, confusionMat, precision, recall, f1Score] = TestTrafficSignsCNN(testPath, img_size)
    % Tests the trained CNN model on new images and evaluates performance
    if nargin < 1 || isempty(testPath)
        testPath = "Data";  
    end
    if nargin < 2 || isempty(img_size)
        img_size = 40;
    end
    basePath = testPath;
    % Load trained SVM model
    if exist('TrafficSignCNN.mat', 'file')
        load('TrafficSignCNN.mat', 'net');
    else
        error('Trained model not found. Run TrainTrafficSignsCNN first.');
    end

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
        Img = reshape(Img, [img_size, img_size, 1]); % reshape for CNN



        % Predict the label using CNN
        predictedLabel = classify(net, Img);
        predictedLabel = double(string(predictedLabel)); % Convert categorical to numeric

        % Store labels
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

    % Calculate accuracy
    accuracy = correct / n;
    fprintf('Accuracy: %.2f%%\n', accuracy * 100);

    % Compute confusion matrix
    confusionMat = confusionmat(trueLabels, predictedLabels);
    fprintf('Confusion Matrix:\n');
    disp(confusionMat);

    % Plot confusion matrix heatmap
    figure;
    imagesc(confusionMat);
    colormap('hot');
    colorbar;
    xlabel('Predicted Labels');
    ylabel('True Labels');
    title('CNN Confusion Matrix Heatmap');

    % Calculate Precision, Recall, F1-Score
    precision = diag(confusionMat) ./ sum(confusionMat, 2);
    recall = diag(confusionMat) ./ sum(confusionMat, 1)';
    recall(isnan(recall)) = 0;
    f1Score = 2 * (precision .* recall) ./ (precision + recall);

    % Display Precision, Recall, and F1-score
    for i = 1:length(precision)
        fprintf('Class %d - Precision: %.2f | Recall: %.2f | F1-Score: %.2f\n', ...
            i-1, precision(i), recall(i), f1Score(i));
    end

    % Calculate macro metrics
    macroPrecision = mean(precision, 'omitnan');
    macroRecall = mean(recall, 'omitnan');
    macroF1Score = mean(f1Score, 'omitnan');

    fprintf('Macro-averaged Precision: %.2f\n', macroPrecision);
    fprintf('Macro-averaged Recall: %.2f\n', macroRecall);
    fprintf('Macro-averaged F1-Score: %.2f\n', macroF1Score);

    fprintf('CNN Testing completed.\n');
end