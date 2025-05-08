function TrainTrafficSignsCNN(trainPath, img_size)
    if nargin < 1 || isempty(trainPath)
        trainPath = "Data";  
    end
    if nargin < 2 || isempty(img_size)
        img_size = 40;
    end
    basePath = trainPath; 
    
    images = [];
    labels = [];

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

        if isempty(images)
            images(:,:,:,1) = Img; 
            labels(1, 1) = ClassID; 
        else
            images(:,:,:,end+1) = Img;
            labels(end+1, 1) = ClassID; 
        end
   
    end
    labels = categorical(labels);

    TrainCNN(images, labels, img_size);
end


function TrainCNN(images, labels, img_size)
    layers = [
        imageInputLayer([img_size img_size 1])
        
        convolution2dLayer(3,8,'Padding','same')
        batchNormalizationLayer
        reluLayer

        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,16,'Padding','same')
        batchNormalizationLayer
        reluLayer

        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,32,'Padding','same')
        batchNormalizationLayer
        reluLayer

        fullyConnectedLayer(43)
        softmaxLayer
        classificationLayer];

    options = trainingOptions('adam', ...
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',20, ...
        'MiniBatchSize',128, ...
        'Shuffle','every-epoch', ...
        'ValidationFrequency',30, ...
        'Verbose',true); %, ...
        %'Plots','training-progress');

    net = trainNetwork(images, labels, layers, options);

    save('TrafficSignCNN.mat', 'net');
    fprintf('CNN training completed. Model saved as TrafficSignCNN.mat\n');
end
