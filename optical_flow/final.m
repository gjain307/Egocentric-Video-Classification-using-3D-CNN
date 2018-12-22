clear all;
close all;
clc;
addpath('mex');
mypath = 'C:\Users\Dell\Desktop\im2flow\output';
for k = 1:4
  jpgFilename1 = strcat('P04_', num2str(k), '.jpg');
  imageData1 = imread(jpgFilename1);
  jpgFilename2 = strcat('P04_', num2str(k+1), '.jpg');
  imageData2 = imread(jpgFilename2);
   im1 = im2double(imageData1);
   im2 = im2double(imageData2);

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
imwrite(imflow,['flow4_', num2str(k),'.jpg']);
drawnow; % Force display to update immediately.
end