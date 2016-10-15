%% Group 15 - Demonstration
db = imageSet('att_faces', 'Recursive');

%% Display montage of first face
figure;
montage(db(1).ImageLocation);
title('Images of Person 1');

%%  Display Query Image and Database Side-Side
personToQuery = 1;
galleryImage = read(db(personToQuery),1);
figure;

for i=1:size(db,2)
    imageList(i) = db(i).ImageLocation(5);
end
subplot(1,2,1);
imshow(galleryImage);
subplot(1,2,2);
montage(imageList);


%% Split Database into Training & Test Sets
[training, test] = partition(db, [0.8, 0.2]);


%% Extract and display Histogram of a single face
person = 1;
[hogFeature, vis] = ...
    extractHOGFeatures(read(training(person),1));
figure;
subplot(2, 1, 1);
imshow(read(training(person),1));
title('Input Face');
subplot(2, 1, 2);
plot(vis);
title('HoG Feature');

%% Train
trainingFeatures = zeros(size(training,2)*training(1).Count,4680); % (320, 4680)
trainingLabel = cell(1, size(trainingFeatures, 1));
personIndex = cell(1, size(training, 2));
featureCount = 1;

% for each folder in training (out of 40 folders)
for i=1:size(training,2)
    % for each image in training (out of 8 pictures)
    for j = 1:training(i).Count
        trainingFeatures(featureCount,:) = extractHOGFeatures(read(training(i),j)); % HoG for each image in the folder
        trainingLabel{featureCount} = training(i).Description;    
        featureCount = featureCount + 1;
    end
    personIndex{i} = training(i).Description;
end

%% Create 40 class classifier using fitcecoc 
faceClassifier = fitcecoc(trainingFeatures,trainingLabel);

%% Test Images from Test Set 
person = 1;
queryImage = read(test(person),1);
queryFeatures = extractHOGFeatures(queryImage);
personLabel = predict(faceClassifier,queryFeatures);
% Map back to training set to find identity 
booleanIndex = strcmp(personLabel, personIndex);
integerIndex = find(booleanIndex);
subplot(1,2,1);imshow(queryImage);title('Query Face');
subplot(1,2,2);imshow(read(training(integerIndex),1));title('Matched Class');

%% Test First 5 People from Test Set
figure;
figureNum = 1;
for person=1:5
    for j = 1:test(person).Count
        queryImage = read(test(person),j);
        queryFeatures = extractHOGFeatures(queryImage);
        personLabel = predict(faceClassifier,queryFeatures);
        % Map back to training set to find identity
        booleanIndex = strcmp(personLabel, personIndex);
        integerIndex = find(booleanIndex);
        subplot(2,2,figureNum);imshow(imresize(queryImage,3));title('Query Face');
        subplot(2,2,figureNum+1);imshow(imresize(read(training(integerIndex),1),3));title('Matched Class');
        figureNum = figureNum+2;
        
    end
    figure;
    figureNum = 1;

end
