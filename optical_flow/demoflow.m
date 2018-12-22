addpath('mex');

myFolder = 'C:\Users\Dell\Desktop\im2flow\input';
mypath = 'C:\Users\Dell\Desktop\im2flow\output';
% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end
% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(myFolder, '*.jpg'); % Change to whatever pattern you need.

for k = 1 : length(theFiles)-1
  baseFileName1 = theFiles(k).name;
  baseFileName2 = theFiles(k+1).name;
  fullFileName1 = fullfile(myFolder, baseFileName1);
  fullFileName2 = fullfile(myFolder, baseFileName2);
  fprintf(1, 'Now reading %s\n', fullFileName1);
  fprintf(1, 'Now reading %s\n', fullFileName2);
   % Now do whatever you want with this file name,
  % such as reading it in as an image array with imread()
  imageArray1 = imread(fullFileName1);
  imageArray2 = imread(fullFileName2);

% load the two frames
im1 = im2double(imageArray1);
im2 = im2double(imageArray2);

% im1 = imresize(im1,0.5,'bicubic');
% im2 = imresize(im2,0.5,'bicubic');

% set optical flow parameters (see Coarse2FineTwoFrames.m for the definition of the parameters)
alpha = 0.012;
ratio = 0.75;
minWidth = 20;
nOuterFPIterations = 7;
nInnerFPIterations = 1;
nSORIterations = 30;

para = [alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations];

% this is the core part of calling the mexed dll file for computing optical flow
% it also returns the time that is needed for two-frame estimation
tic;
[vx,vy,warpI2] = Coarse2FineTwoFrames(im1,im2,para);
toc



% output gif
clear volume;
volume(:,:,:,1) = im1;
volume(:,:,:,2) = im2;
if exist('output','dir')~=7
    mkdir('output');
end
%frame2gif(volume,fullfile('output',[example '_input.gif']));
%volume(:,:,:,2) = warpI2;
%frame2gif(volume,fullfile('output',[example '_warp.gif']));


% visualize flow field
clear flow;
flow(:,:,1) = vx;
flow(:,:,2) = vy;
imflow = flowToColor(flow);
imwrite(imflow,[mypath,num2str(k),'.jpg']);

end
