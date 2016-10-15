%% Group 15 - Demonstration
db = imageSet('images', 'Recursive');

%% check the faces are there
figure;
montage(db(3).ImageLocation);

%%
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

training_rows = 0;
for i=1:size(training, 2)
   training_rows = training_rows + training(i).Count;
end
trainingFeatures = zeros(training_rows, 4680);
trainingLabel = cell(1, size(trainingFeatures, 1));
personIndex = cell(1, size(training, 2));
featureCounter = 1;

for i=1:size(training, 2)
    
    for j=1:training(i).Count
       trainingFeatures(featureCounter, :) = extractHOGFeatures(read(training(i), j));
       trainingLabel{featureCounter} = training(i).Description;
       featureCounter = featureCounter + 1;
    end
    
    personIndex{i} = training(i).Description;
    
end
%%
faceClassifier = fitcecoc(trainingFeatures,trainingLabel);


%% Test Images from Test Set 
person = 36;
queryImage = read(test(person),1);
queryFeatures = extractHOGFeatures(queryImage);
personLabel = predict(faceClassifier,queryFeatures);
% Map back to training set to find identity 
booleanIndex = strcmp(personLabel, personIndex);
integerIndex = find(booleanIndex);
subplot(1,2,1);imshow(queryImage);title('Query Face');
subplot(1,2,2);imshow(read(training(integerIndex),1));title('Matched Class');


%%
peopleToTest = [36, 37, 38, 39]
for i=1:size(peopleToTest, 2)
    person = peopleToTest(i)
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

end