import os

import matplotlib.pyplot as plt
import numpy as np
    
conditions = [
    '8 nm ; 400 ns',    
    '8 nm ; 1600 ns',    
    '4 nm ; 400 ns',    
    '4 nm ; 1600 ns',]    
    
result_paths = [
    '8nm_400ns.txt',
    '8nm_1600ns.txt',
    '4nm_400ns.txt',
    '4nm_1600ns.txt']
    
def readResults(p):
    with open(p, 'r') as f:
        lines = f.readlines()
        lines = [line.split('\t') for line in lines]
        lines = [[line[1], line[3], line[5], line[7]] for line in lines]
        lines = [[float(x) for x in line] for line in lines]
    return lines

all_a = []
all_b = []
all_c = []

for result_path in result_paths:
    results = readResults(result_path)
    ids = [line[0] for line in results]
    a = [line[1] for line in results]
    b = [line[2] for line in results]
    c = [line[3] for line in results]
    
    all_a.append(a)
    all_b.append(b)
    all_c.append(c)
    
all_a = np.array(all_a)    
all_b = np.array(all_b)    
all_c = np.array(all_c)    
print(all_a)


# plot mean,std of a
for variable in ['a', 'b', 'c']:
    fig, ax = plt.subplots()
    all_variable = vars()['all_' + variable]
    ax.errorbar(
        range(len(all_variable)),
        [np.mean(a) for a in all_variable],
        [np.std(a) for a in all_variable],
        linestyle = 'None',
        fmt = 'o')

    ax.set_title(
        'Scan correction fit y = a*exp(-bx)+c'
        '\n Mean and std for all beams in mFOV 0')
    ax.set_ylabel(variable, fontsize = 30)
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(conditions)
    for tick in ax.get_xticklabels():
        tick.set_rotation(55)
    fig.savefig(variable + '.jpg', bbox_inches='tight')

plt.show()