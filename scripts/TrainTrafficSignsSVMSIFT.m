function TrainTrafficSignsSVMSIFT(trainPath, img_size)
    if nargin < 1 || isempty(trainPath)
        trainPath = "Data";  
    end
    if nargin < 2 || isempty(img_size)
        img_size = 40;
    end
    basePath = trainPath; 
    
    % Initialize storage for features and labels
    trainingData = [];
    trainingLabels = [];

    trainingFile = readtable(trainPath + "\Train.csv");
  
    n = size(trainingFile, 1);
    for i = 1:n
        imgPath = trainingFile.Path(i);
        fileName = split(imgPath{1}, "/");
        fileName = fileName{end};
        
        ImgFile = fullfile(basePath, imgPath); 
        ClassID = trainingFile.ClassId(i);
        RoiX1 = trainingFile.Roi_X1(i);
        RoiY1 = trainingFile.Roi_Y1(i);
        RoiX2 = trainingFile.Roi_X2(i);
        RoiY2 = trainingFile.Roi_Y2(i);
        Img = imread(ImgFile);
        
        fprintf(1, 'Currently training: %s Class: %d Sample: %d / %d\n', fileName, ClassID, i, n);

        Img = Img(RoiY1 + 1:RoiY2 + 1, RoiX1 + 1:RoiX2 + 1);

        % Resize the image to a fixed size
        Img = imresize(Img, [img_size img_size]);

        % Detect SIFT keypoints
        keypoints = detectSIFTFeatures(Img);
        
        % Extract SIFT features
        [features, ~] = extractFeatures(Img, keypoints);
        features = mean(features, 1); 
        trainingData = [trainingData; features];    
        trainingLabels = [trainingLabels; ClassID];
   
    end

    % Train SVM model
    TrainSVM(trainingData, trainingLabels);
end

   
   
function TrainSVM(trainingData, trainingLabels)
    % Trains an SVM classifier and saves the model.

    fprintf('Training SVM model...\n');
    SVMModel = fitcecoc(trainingData, trainingLabels, 'Coding', 'onevsone');
    
    save('TrafficSignSVMSIFT.mat', 'SVMModel');
    fprintf('SVM training completed. Model saved as TrafficSignSVM.mat\n');
end

