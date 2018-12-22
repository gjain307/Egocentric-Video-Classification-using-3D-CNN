import os
import numpy as np

def class_labels_from_dir_structure(directory, get_count=False):
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
        print "\t{} images in {}".format(num_ex, os.path.basename(path))
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

    if get_count==True:
        return class_labels, tot_num_examples
        
    return class_labels


def copy_dir_structure(inputpath, outputpath):
    """Copy the directory structure of a folder to another"""
    for dirpath, dirnames, filenames in os.walk(inputpath):
        structure = os.path.join(outputpath, dirpath[len(inputpath):])
        if not os.path.isdir(structure):
            os.mkdir(structure)

def next_file(inputs, ext):
    """Yields the path of next file of the given extension in a folder"""
    for file in os.listdir(inputs):
        if file.endswith("." + ext):
            yield (os.path.join(inputs, file))

# def walklevel(some_dir, level=1):
#     """ From https://stackoverflow.com/questions/229186/\
#     os-walk-without-digging-into-directories-below
#     Walk directories upto given level"""
#     some_dir = some_dir.rstrip(os.path.sep)
#     assert os.path.isdir(some_dir)
#     num_sep = some_dir.count(os.path.sep)
#     for root, dirs, files in os.walk(some_dir):
#         yield root, dirs, files
#         num_sep_this = root.count(os.path.sep)
#         if num_sep + level <= num_sep_this:
#             del dirs[:]

def subdir(d):
    """Get subdirectories in a goven folder"""
    return [os.path.join(d,o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]