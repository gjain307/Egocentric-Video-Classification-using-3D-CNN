% Read files file1.txt through file20.txt, mat1.mat through mat20.mat
% and image1.jpg through image20.jpg.  Files are in the current directory.
for k = 1:3535
  jpgFilename1 = strcat('pizza3_', num2str(k), '.jpg');
  imageData1 = imread(jpgFilename1);
  jpgFilename2 = strcat('pizza3_', num2str(k+1), '.jpg');
  imageData2 = imread(jpgFilename2);
   obj = imfuse(imageData1,imageData2); %imwrite(obj,'stereo.png');
      imshow(obj)
      imwrite(obj,['fuse3_' num2str(k),'.jpg']);
  drawnow; % Force display to update immediately.
end