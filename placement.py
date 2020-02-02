from utils import *
# from plot import *
import numpy as np
from queue import PriorityQueue as PQ
import copy
import argparse
from tkinter import *
from tkinter import ttk
COLORS = ['black', 'red', 'yellow', 'azure4', 'orange', 'maroon', 'pink', 'lime green', 'dark violet', 'green']


class Placer:
    def __init__(self, infile, verbose=False):
        self.verbose = verbose
        self.grid_size, self.num_cells, self.num_nets, self.nets = parseInput(infile)
        self.blocks = block2Net(self.num_cells, self.nets)

        self.randomPlacement()
        self.startGUI()
        self.plot()

    # GUI
    def startGUI(self, width=1000, height=500, background_color='white'):
        '''
        set up GUI
        '''
        self.root = Tk()
        self.frame = ttk.Frame(self.root, width=width, height=height)
        self.frame.pack()
        self.canvas = Canvas(self.frame, bg=background_color, width=width, height=height)
        self.canvas.pack()

        # randomPlacement
        self.random_button = ttk.Button(self.frame, text="Random Placement", command=self.showRandomPlacement)
        self.random_button.pack()

    # modify for generic plotting function
    def plot(self, width=1600, height=1000):
        x_bound, y_bound = self.grid_size
        sizex = width / x_bound
        sizey = height / y_bound

        for i in range(x_bound):
            for j in range(y_bound):
                self.canvas.create_rectangle(sizex * i, sizey * j, sizex * (i + 1), sizey * (j + 1), fill='white')

                if self.state[i, j] != -1:
                    self.canvas.create_text(sizex * (i + 0.5), sizey * (j + 0.5), text=self.state[i, j])

    def showRandomPlacement(self):
        ''' Reset the GUI to initial state'''
        self.randomPlacement()
        self.plot()
        print('reset')

    def randomPlacement(self):
        ''' random placement'''
        # place all blocks in order
        ordered = np.concatenate([np.arange(self.num_cells), -1 * np.ones(np.prod(self.grid_size) - self.num_cells, dtype=np.int32)])
        # random shuffling
        self.state = np.random.permutation(ordered).reshape(self.grid_size)
        print(self.state)
        return self.state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CPEN 513 Assignment 2: Placement')
    parser.add_argument('--infile', '-i', default='benchmarks/cm138a.txt', help='input file')  # yaml
    args = parser.parse_args()

    placer = Placer(args.infile)
    placer.root.mainloop()
