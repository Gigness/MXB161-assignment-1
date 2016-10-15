%% Group 15 - Demonstration
db = imageSet('group_faces', 'Recursive');

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

