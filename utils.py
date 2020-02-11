import argparse
import numpy as np


epsilon = 1e-4


def parseInput(infile):
    '''
    Take an input file, parse the grid size, total number of cells,
    and generate a net2block mapping (i.e. net ID -> block IDs)
    '''
    with open(infile, 'r') as f:
        num_blocks, num_nets, ny, nx = tuple([int(x) for x in f.readline().rstrip('\n').split(' ')])
        nets = []
        for _ in range(num_nets):
            f.readline()  # skip empty line
            nets.append([int(cell) for i, cell in enumerate(f.readline().rstrip('\n').split(' ')) if cell != '' and i != 0])

        return (ny, nx), num_blocks, num_nets, nets


def block2Net(num_blocks, nets):
    '''
    This function takes a net2block mapping, and convert to a block2net mapping (i.e. block ID -> net IDs)
    '''
    blocks = []
    for i in range(num_blocks):
        block = []
        for j in range(len(nets)):
            if i in nets[j]:
                block.append(j)
        blocks.append(block)
    return blocks


def getBBox(input):
    '''
    Evaluate half-perimeter of the smallest bounding box of n coordinate
    input: a numpy array with size n by 2
    '''
    min = input.min(axis=0)
    max = input.max(axis=0)
    return min, max


def evalCost(input):
    '''
    Evaluate half-perimeter of the smallest bounding box of n coordinate
    input: a numpy array with size n by 2
    '''
    min, max = getBBox(input)
    return np.sum(max - min)


def betterEvalDeltaCost(old, new):
    '''
    old on the boundary, reevaluate the boundary
    else, see if new become new boundary
    '''
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CPEN 513 Assignment 2: Placement')
    parser.add_argument('--infile', '-i', default='benchmarks/cm138a.txt', help='input file')
    args = parser.parse_args()
    grid_size, num_blocks, num_nets, nets = parseInput(args.infile)
    print('grid size: ', grid_size)
    print('num_blocks: ', num_blocks)
    print('nets: ', nets)
    blocks = block2Net(num_blocks, nets)
    print('blocks: ', blocks)
