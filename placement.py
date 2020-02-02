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
        self.cost = 0

        self.startGUI()
        self.randomPlacement()
        self.plot()

    # GUI
    def startGUI(self, width=1400, height=600, background_color='white'):
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

        # current Cost
        self.cost_var = StringVar()
        self.cost_label = ttk.Label(self.frame, textvariable=self.cost_var)
        self.cost_label.pack()

    # modify for generic plotting function
    def plot(self, width=1400, height=600):
        y_bound, x_bound = self.grid_size
        sizex = width / x_bound
        sizey = height / y_bound

        for i in range(y_bound):
            for j in range(x_bound):
                self.canvas.create_rectangle(sizex * j, sizey * i, sizex * (j + 1), sizey * (i + 1), fill='white')

                # Fixme: variable font size based on number of cells
                if self.state[i, j] != -1:
                    self.canvas.create_text(sizex * (j + 0.5), sizey * (i + 0.5), text=self.state[i, j], font=('Arial', 10))

    def showRandomPlacement(self):
        ''' Reset the GUI to initial state'''
        self.randomPlacement()
        self.plot()
        # print('reset')

    def randomPlacement(self):
        ''' random placement'''
        # place all blocks in order
        ordered = np.concatenate([np.arange(self.num_cells), -1 * np.ones(np.prod(self.grid_size) - self.num_cells, dtype=np.int32)])
        # random shuffling
        self.state = np.random.permutation(ordered).reshape(self.grid_size)
        # print(self.state)
        self.cost = self.evalTotalCost()
        self.cost_var.set([self.cost, np.sum(self.cost)])
        return self.state

    def evalTotalCost(self):
        self.block2Coordinates()
        # print(self.block_to_coordinates)
        cost = np.zeros(self.num_nets, dtype=np.int32)
        for i in range(self.num_nets):
            # print(self.nets[i])
            cost[i] = self.evalCost(self.block_to_coordinates[self.nets[i], :])
        # print('cost of each net', cost)
        # print('current total cost', np.sum(cost))
        return cost

    def evalCost(self, input):
        min = input.min(axis=0)
        max = input.max(axis=0)
        return np.sum(max - min)

    def block2Coordinates(self):
        y_bound, x_bound = self.grid_size
        self.block_to_coordinates = np.zeros((self.num_cells, 2))

        for i in range(y_bound):
            for j in range(x_bound):
                current_cell = self.state[i, j]
                if current_cell != -1:
                    self.block_to_coordinates[current_cell] = (i, j)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CPEN 513 Assignment 2: Placement')
    parser.add_argument('--infile', '-i', default='benchmarks/cm138a.txt', help='input file')  # yaml
    args = parser.parse_args()

    placer = Placer(args.infile)
    placer.root.mainloop()
