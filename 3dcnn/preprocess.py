import cv2
import imageio
import os
import numpy as np
import sys
from glob import glob


def load_images_from_folder(folder, get_count=False):
    images = []
    count = 0
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            img = cv2.resize(img, (112, 112))
            images.append(img)
            if get_count==True:
                count += 1
    if get_count==True:
        return images, count
    return images


def walklevel(some_dir, level=1): 
    some_dir = some_dir.rstrip(os.path.sep)
    #remove the trailing '/' from the path if any
    assert os.path.isdir(some_dir)
    #check whether the directory exists or not
    num_sep = some_dir.count(os.path.sep)
    #find out the number of seperating '/'
    for root, dirs, files in os.walk(some_dir, topdown=True):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


def frames_to_arr(frames_dir, img_type=".png"):
    """
    Return frames of videos stored in directory as numpy arrays.
    Directory structure should be like:
    Train/
        /Angry
            /00324234/*.jpg
            /03423435/*.jpg
        /Sad
            /42342342/*.jpg
            /23434213/*.jpg
        ...
    """

    # Get class labels from direcotry structure
    itr = iter(walklevel(frames_dir, level=1))
    _, dirs, _ = itr.next()
    #return the list of all directories inside the frames_dir

    # Get number of classes
    num_classes = len(dirs)
    print "[Info] Number of classes: {}".format(num_classes)

    # Get number of examples in each class
    num_examples = []
    output = []
    
    for path, dirs, _ in itr:
        #path gives the path inside the frame_dir and dirs gives the dirs inside the individual dirs
        num_ex = len(dirs)
        # num_ex in our case is 12 since there are 12 sequences of each subject
        num_examples.append(num_ex)
        # print "\t{} videos in class {}".format(num_ex, os.path.basename(path))
        num_videos = 0

        for frame_folder in dirs:
            frame_folder = os.path.join(path, frame_folder)
            vid, num_frames = load_images_from_folder(frame_folder, get_count=True)
           
            if num_frames > 16:
                vid = vid[:16-num_frames]
            
            output.append(vid)
            num_videos += 1
            sys.stdout.write("\r[Info] {} videos processed in {}".format(num_videos, os.path.basename(path)))
            sys.stdout.flush()
        print ""
        # print path
    
    tot_num_examples = sum(num_examples)
    print "\tTotal number of examples: {}".format(tot_num_examples)
    
    # Create class labels
    class_labels = np.ones(tot_num_examples)
    ind = 0
    start = 0
    end = 0
    for num_cl_ex in num_examples:
        end = start + num_cl_ex
        class_labels[start:end] = ind
        start = end
        ind += 1


    #### Get video frames

    # Get all videos in directory
    # result = [y for x in os.walk(frames_dir) for y in glob(os.path.join(x[0], img_type))]
    

    # for frame_path in result:
        

    print '\n'
    
    # Convert to numpy array
    output = np.array(output)

    print "Output shape: {}".format(output.shape)
    print "Labels shape: {}".format(class_labels.shape)

    
    # Subtract mean and divide by maximum value
    output = output.astype('float32')
    output -= np.mean(output)
    output /= np.max(output)

    # Save processed videos to disk
    # np.save(data_save_path, output)
    # np.save(labels_save_path, class_labels)

    return output, class_labels


def class_labels_from_dir_structure(directory):
    itr = iter(os.walk(directory))
    _, dirs, _ = itr.next()

    # Get number of classes
    num_classes = len(dirs)
    print "[Info] Number of classes: {}".format(num_classes)

    # Get number of ecamples in each class
    num_examples = []
    
    for path, dirs, files in itr:
        num_ex = len(files)
        num_examples.append(num_ex)
        print "\t{} videos in class {}".format(num_ex, os.path.basename(path))
        # print path
    
    tot_num_examples = sum(num_examples)
    print "\tTotal number of examples: {}".format(tot_num_examples)
    
    # Create class labels
    class_labels = np.ones(tot_num_examples)
    ind = 0
    start = 0
    end = 0
    for num_cl_ex in num_examples:
        end = start + num_cl_ex
        class_labels[start:end] = ind
        start = end
        ind += 1

    return class_labels


def vid_to_arr(vid_dir, video_type='*.avi', output_name='video_array.npy', debug=False):
    '''
    Recursively process videos in the given directory. Converts
    videos to 3D numpy arrays.
    '''
    # Get class labels from directory structure
    labels = class_labels_from_dir_structure(vid_dir)

    # Path where processed file is to be retrieved from/ saved to
    save_path = os.path.join(vid_dir,os.path.basename(vid_dir))
    data_save_path = save_path + '_data.npy'
    labels_save_path = save_path + '_labels.npy'
    
    # If processed file available, just load from it
    if os.path.isfile(data_save_path)==True:   

        print("[Info] Loading")
        
        opt = np.load(data_save_path)
        labels = np.load(labels_save_path)
        
        print("\tOutput shape: {}\n".format(opt.shape))
        print("\tLabels shape: {}\n".format(labels.shape))

        # Center crop
        # output = opt[:, :, 8:120, 30:142, :] # (num_ex, l, h, w, c)

    # Else generate processed file and save to disk for future use
    else:

        # Get all videos in directory
        result = [y for x in os.walk(vid_dir) for y in glob(os.path.join(x[0], video_type))]
        output = []
        num_videos = 0

        for video_path in result:
            # Capture video
            reader = imageio.get_reader(video_path)

            # Get individual frames
            vid = []
            num_frames = len(reader)
            
            # mid = int(num_frames/2)
            # print num_frames
            img = []

            for i in range(num_frames):
                # img = cv2.resize(reader.get_data(i), (171, 128))
                img = cv2.resize(reader.get_data(i), (112, 112))
                vid.append(img)

            if(num_frames <= 16):
                for i in range(16 - num_frames):
                    vid.append(img)
            else:
                vid = vid[:16-num_frames]
            
            output.append(vid)
            num_videos += 1
            sys.stdout.write("\r[Info] {} videos processed".format(num_videos))
            sys.stdout.flush()

        print '\n'
        
        # Convert to numpy array
        output = np.array(output)
        
        # Subtract mean and divide by maximum value
        output = output.astype('float32')
        output -= np.mean(output)
        output /= np.max(output)

        print "Output shape: {}".format(output.shape)
        print "Labels shape: {}".format(labels.shape)

        # Save processed videos to disk
        np.save(data_save_path, output)
        np.save(labels_save_path, labels)

    return output, labels



def main():
    data_dir_path = 'data/afew_2016_modified/Train/Fear'
    vid_to_arr(data_dir_path, debug=True)
    

if __name__ == '__main__':
    main()