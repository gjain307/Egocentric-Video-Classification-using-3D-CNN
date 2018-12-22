import cv2
import imageio
import os
import numpy as np
import sys
from glob import glob

def class_labels_from_dir_structure(directory):
    itr = iter(os.walk(directory))
    print "[Info] Reading data directory"
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

    # Get all videos in directory
    result = [y for x in os.walk(vid_dir) for y in glob(os.path.join(x[0], video_type))]
    output = []
    num_videos = 0

    for video_path in result:
        # Capture video
        cap = cv2.VideoCapture(video_path)
        # cap.set(cv2.CAP_PROP_FPS, 200)
        # test = cap.get(cv2.CAP_PROP_FPS)
        # print test
        # Get individual frames
        vid = []
        k= 0
        img = np.zeros([171,128])
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            print k
            cap.grab()
            ret, img = cap.retrieve()
            img = cv2.resize(img, (171, 128))
            cv2.imwrite('test_out/test{}.jpg'.format(k), img)
            # print img
            if not ret:
                break
            vid.append(img)
            k += 1
        vid = np.array(vid, dtype=np.float32)
        if(debug==True):
            pass
            # print("Number of frames in video {}: {}".format(os.path.basename(video_path), cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            # print("FPS of video: {}".format(cv2.CAP_PROP_FPS))
            # print vid.shape
            # print vid

        # Extract frames for C3D network
        # start_frame = 0
        # if (len(vid) < 3):
            # pass
            # print "[Error] Video {} has less than 3 frames".format(os.path.basename(video_path))
        # else:
            # X = vid[start_frame:(start_frame + 10), :, :, :]
            # print X.shape
        
        # Subtract mean
        # mean_cube = np.load('models/train01_16_128_171_mean.npy')
        # mean_cube = np.transpose(mean_cube, (1, 2, 3, 0))
        # X -= mean_cube

        # Center crop
        # X = X[:, 8:120, 30:142, :] # (l, h, w, c)
        cap.release()
        output.append(vid)
        print len(vid)
        num_videos += 1
        sys.stdout.write("\r[Info] {} videos processed".format(num_videos))
        sys.stdout.flush()

    print '\n'
    # Convert to numpy array and save to disk
    output = np.array(output)
    # print output[0].shape
    # print "\n[Info] Saving model to file.."
    # np.save(open(output_name, 'w'), output)
    # print "[Info] Done"

    return output, labels



def main():
    data_dir_path = 'data/afew_2016_modified/Train/Fear'
    vid_to_arr(data_dir_path, debug=True)
    

if __name__ == '__main__':
    main()