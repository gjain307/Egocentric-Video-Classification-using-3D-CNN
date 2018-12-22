% Specify the folder where the files live.
myFolder = 'C:\Users\Dell\Desktop\summer_\frames';
mypath = 'C:\Users\Dell\Desktop\summer_\input';
% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end
% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(myFolder, '*.jpg'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
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
  obj = imfuse(imageArray1,imageArray2); %imwrite(obj,'stereo.png');
      imshow(obj)
      imwrite(obj,[mypath,num2str(k),'.jpg']);
  drawnow; % Force display to update immediately.
  
end