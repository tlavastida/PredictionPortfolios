# Utility functions

import random
import mmap
import numpy as np

from MinWeightPerfectMatching import MinWeightPerfectMatching

# computes jaccard similarity between matchings M1 and M2
def JaccardMatching(M1,M2):
    intersection = sum(1 if e in M2 else 0 for e in M1)
    union = len(M1) + len(M2) - intersection
    return intersection / union

# median of a list L
# if |L| is odd, returns middle element,
# it |L| is even, returns
def median(L):
    n = len(L)
    if n % 2 == 0:
        idx = n // 2 - 1
    else:
        idx = n // 2
    
    return sorted(L)[idx]

# takes coordinate-wise median of the duals
# to aggregate into one set of duals
def median_duals(dual_list):
    V = dual_list[0].keys()
    duals = {v:0 for v in V}
    for v in V:
        L = [d[v] for d in dual_list]
        duals[v] = median(L)

    return duals

# takes coordinate-wise average of duals
# to aggregate
def average_duals(dual_list):
    V = dual_list[0].keys()
    duals = {v:0 for v in V}
    for v in V:
        avg = sum(d[v] for d in dual_list) / len(dual_list)
        # randomly round up or down by the fractional part
        # e.g. 1.75 -> 2 with probability 0.75 and 1 with probability 0.25
        int_part = int(avg)
        frac_part = avg - int_part
        if avg < 0:
            rounding = -1 if random.random() <= abs(frac_part) else 0
        else:
            rounding = 1 if random.random() <= frac_part else 0
        
        duals[v] = int_part + rounding

    return duals

# instance_gen is a function that returns some
# randomly drawn instance from a distribution
# over instances
# this trains duals for this distribution
# using samples
def train_duals(samples, instance_gen):
    dual_list = []
    for i in range(samples):
        print('Running training sample {}'.format(i))
        G = instance_gen()
        _, d = MinWeightPerfectMatching(G) # pylint: disable=unbalanced-tuple-unpacking
        dual_list.append(d)
    print("Aggregating duals")
    return median_duals(dual_list)

def print_dict_to_file(d,fname):
    f = open(fname,'w')
    for k in d.keys():
        f.write('{} : {}\n'.format(k,d[k]))
    f.close()

# code for reading data files for clustering

# Returns a list of the newline locations of the input file
def generate_newline_locations(filename):
    # Open the file as a mmap (bytes) to do efficient things counting
    file = open(filename,"r+")
    file_map = mmap.mmap(file.fileno(),0)

    # iterate through the file for two reasons:
    # 1) to get how many things are actually in the file
    # 2) to record where the newline characters are in the file into a newline_locs file
    newline_locs = [0]
    while file_map.readline():
        newline_locs.append(file_map.tell())
    del newline_locs[-1]
    file_map.close()
    return newline_locs

# load lines from a file randomly, input requires set of newline_locs and filename
def load_random_lines(filename, newline_locs,lines):
    # First, figure out the format of the file, if it is a csv it must be split at commas, etc etc.
    name, extension = filename.split(".") # pylint: disable=unused-variable
    if extension=="csv":
        delim = b','
    else:
        delim = None

    # Open the file as a mmap (bytes) to do efficient processing
    file = open(filename,"r+")
    file_map = mmap.mmap(file.fileno(),0)

    # generate a random permutation of line indices to grab
    # print(len(newline_locs), " ", lines)
    samples = np.random.choice(len(newline_locs),lines,replace=False)

    # definies a temporary function which, when given the line number to get, 
    # seeks for the correct byte to read the next line and convert to a nparray
    def get_nparray_for_line(linenum):
        file_map.seek(newline_locs[linenum])
        line = file_map.readline()
        return np.array([float(y) for y in line.split(delim)])

    # list comprehension to generate the result to return
    results = [get_nparray_for_line(sample) for sample in samples]
    return results