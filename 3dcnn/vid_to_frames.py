from ffmpy import FFmpeg, FFprobe
import subprocess
import argparse
import utils
import sys
import os

def video_duration(vid_path):
    """ Get duration of video in seconds """

    # Build ffprobe command
    inputs={vid_path:None}
    options = '-show_entries stream=r_frame_rate,nb_read_frames -select_streams v -count_frames -of compact=p=0:nk=1 -v 0'
    ff = FFprobe(global_options=options, inputs=inputs)
    get_dur = ff.cmd

    # Run the ffprobe command in  a subprocess
    result = subprocess.check_output(get_dur.split())
    result = result.rstrip()    # Strip newline character from output 
    num_frames = float(result.split('|')[1])    # Since output of command is like "25/1|69"
    frame_rate = result.split('|')[0].split('/')

    vid_duration = num_frames/(float(frame_rate[0])/float(frame_rate[1]))
    return vid_duration


def vid_to_frames(inputs='.', input_format='avi', output_dir='output', output_format='jpg', frames_per_video=10):
    """Extract given number of frames evenly spaced from video"""
    
    if(os.path.isdir(inputs)==False):
        print "[Info] Processing video"
        dur = video_duration(inputs)
        options = "-nostats -loglevel 0 -r {}".format(float(frames_per_video)/float(dur))
        
        file_name=os.path.splitext(os.path.basename(inputs))[0]
        inputs={inputs:None}
        
        output = "{}/{}%03d.{}".format(output_dir, file_name, output_format)
        outputs = {output: options}
        ff = FFmpeg(inputs=inputs,outputs=outputs)
        print "\nCOMMAND:" + ff.cmd
     
    else:
        print "[Info] Processing video directory {}".format(inputs)
        print "[Info] Writing to directory: {}/{}".format(os.getcwd(), output_dir)
        utils.copy_dir_structure(inputs, output_dir)

        for subdir in utils.subdir(inputs):
            print "\n[Info] Processing {}".format(subdir)
            for subsubdir in utils.subdir(subdir):
                print "\r[Info] Processing {}".format(subsubdir)
                for vid in utils.next_file(subsubdir, input_format):
                    sys.stdout.write("\r[Info] Processing {}".format(os.path.basename(vid)))
                    sys.stdout.flush()
                    # print vid
                    dur = video_duration(vid)
                    # print dur
                    options = '-nostats -loglevel 0 -r {}'.format(float(frames_per_video)/float(dur))        
                    dir_name=os.path.join(output_dir, vid[len(inputs):-4])
                    if not os.path.isdir(dir_name):
                        os.mkdir(dir_name)
                    file_name = os.path.basename(dir_name)
                    # print file_name
                    input_dict={vid:None}
                    # output = "{}_%02d.{}".format(os.path.splitext(dir_name)[0], output_format)
                    output = "{}/%02d.{}".format(dir_name, output_format)
                    outputs={output:options}
                    ff = FFmpeg(inputs=input_dict,outputs=outputs)
                    # print "\nCOMMAND:" + ff.cmd
                    ff.run()



def main():
    # parser = argparse.ArgumentParser(description='Video to frames converter. Converts the given video \
    #     or videos in given folder to the specified number of frames. The output directory structure \
    #     will be the same as the input. Requires ffmpeg installed.')
    # parser.add_argument("-i", "--input_folder", required=True, help="Specify the input folder.")
    # parser.add_argument("-o", "--output_folder", required=True, help="Specify the output folder.")
    # parser.add_argument("-of", "--format", required=True, help="Output format. Should be a valid picture format.")
    # parser.add_argument("-f", "--frames", type=int,  required=True, help="Number of frames to be extracted from a video")
    # args = parser.parse_args()

    # input_folder = args.input_folder
    # output_folder = args.output_folder
    # output_format = args.format
    # frames_per_video = args.frames

    input_folder = 'data/afew_2016_modified/'
    output_folder = 'data/afew_frames'
    output_format = 'jpg'
    frames_per_video = 16
    
    vid_to_frames(inputs=input_folder, output_dir=output_folder, output_format=output_format, frames_per_video=frames_per_video)

if __name__ == '__main__':
    main()