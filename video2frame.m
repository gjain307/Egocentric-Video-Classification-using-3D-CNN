 vid=VideoReader('P04NEW.mp4');
 
 numFrames = vid.NumberOfFrames;
 n = numFrames;
 for i = 0:54000
      frames = read(vid,i+1); 
 imwrite(frames,['P04_' int2str(i+1), '.jpg']);
 im(i+1)=image(frames);
 end
 
 