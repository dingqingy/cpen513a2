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
    '''
    self.nets: List (net ID -> list of block IDs)
    self.blocks: List (block ID -> list of net IDs)
    self.state: np 2D array (coordinates -> block ID)
    self.block_to_coordinates: np array of shape (num_blocks * 2)
    '''

    def __init__(self, infile, verbose=False):
        self.verbose = verbose
        self.grid_size, self.num_blocks, self.num_nets, self.nets = parseInput(infile)
        self.blocks = block2Net(self.num_blocks, self.nets)
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

        # Anneal
        self.anneal_button = ttk.Button(self.frame, text="Anneal", command=self.annealWrapper)
        self.anneal_button.pack()

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

        for net in self.nets:
            for k in range(len(net) - 1):
                start_y, start_x = self.block_to_coordinates[net[k]]
                end_y, end_x = self.block_to_coordinates[net[k + 1]]
                # print('start: {}, end: {}'.format(start, end))
                self.canvas.create_line(sizex * (start_x + 0.5), sizey * (start_y + 0.5), sizex * (end_x + 0.5), sizey * (end_y + 0.5), fill='blue')

    def showRandomPlacement(self):
        ''' Reset the GUI to initial state'''
        self.randomPlacement()
        self.plot()
        # print('reset')

    def randomPlacement(self):
        ''' random placement'''
        # place all blocks in order
        ordered = np.concatenate([np.arange(self.num_blocks), -1 * np.ones(np.prod(self.grid_size) - self.num_blocks, dtype=np.int32)])
        # random shuffling
        self.state = np.random.permutation(ordered).reshape(self.grid_size)
        # print(self.state)
        self.cost = self.evalTotalCost()
        self.cost_var.set(np.sum(self.cost))
        # return self.state

    def evalTotalCost(self):
        self.block2Coordinates()
        # print(self.block_to_coordinates)
        cost = np.zeros(self.num_nets, dtype=np.int32)
        for i in range(self.num_nets):
            # print(self.nets[i])
            cost[i] = evalCost(self.block_to_coordinates[self.nets[i], :])
        # print('cost of each net', cost)
        # print('current total cost', np.sum(cost))
        return cost

    def evalDeltaCost(self, pair):
        '''
        Get Bounding Box for untouched blocks min_other, max_other
        Compare cost between stacked(min_other, max_other, new) and stacked(min_other, max_other, old)
        BTW, is the old net cost available?
        '''
        # print('state:')
        # print(self.state)
        p1, p2 = pair  # coordinate in numpy array
        block_id1, block_id2 = self.state[tuple(p1)], self.state[tuple(p2)]
        # print('block1 id', block_id1)
        proposed_cost = np.zeros(self.num_nets, dtype=np.int32)
        delta = 0

        if block_id1 == -1 and block_id2 == -1:
            print('This should never happen, sth is wrong')
            assert(False)

        elif block_id2 == -1:
            for net_id in self.blocks[block_id1]:
                other = [block for block in self.nets[net_id] if block != block_id1]
                proposed_cost[net_id] = evalCost(np.concatenate([p2.reshape(1, -1), self.block_to_coordinates[other, :]]))
                delta += proposed_cost[net_id] - self.cost[net_id]

        elif block_id1 == -1:
            for net_id in self.blocks[block_id2]:
                other = [block for block in self.nets[net_id] if block != block_id2]
                # print('p1 shape', p1.shape)
                # print('other shape', self.block_to_coordinates[other, :].shape)
                proposed_cost[net_id] = evalCost(np.concatenate([p1.reshape(1, -1), self.block_to_coordinates[other, :]]))
                delta += proposed_cost[net_id] - self.cost[net_id]

        else:
            for net_id in self.blocks[block_id1]:
                # skip delta evalution if both block belong to this net
                if block_id2 not in self.nets[net_id]:
                    proposed = [block if block != block_id1 else block_id2 for block in self.nets[net_id]]
                    # print('proposed blocks')
                    # print(proposed)
                    # min_other, max_other = getBBox(self.block_to_coordinates[other_blocks, :])
                    # TODO: prune if 2 block belong to the same net
                    proposed_cost[net_id] = evalCost(self.block_to_coordinates[proposed, :])
                    delta += proposed_cost[net_id] - self.cost[net_id]
                    # print('proposed_cost for net {} is {}, current cost is {}, current delta is {}'.format(net_id, proposed_cost[net_id], self.cost[net_id], delta))

            # print('block2 id', block_id2)
            for net_id in self.blocks[block_id2]:
                # avoid count the same net again
                if net_id not in self.blocks[block_id1]:
                    proposed = [block if block != block_id2 else block_id1 for block in self.nets[net_id]]
                    # print('proposed blocks')
                    # print(proposed)
                    # min_other, max_other = getBBox(self.block_to_coordinates[other_blocks, :])
                    # TODO: prune evaluated nets?
                    proposed_cost[net_id] = evalCost(self.block_to_coordinates[proposed, :])
                    delta += proposed_cost[net_id] - self.cost[net_id]
                    # print('proposed_cost for net {} is {}, current cost is {}, current delta is {}'.format(net_id, proposed_cost[net_id], self.cost[net_id], delta))

        # print('debug')
        # print('proposed_cost', proposed_cost)
        # print('cost', self.cost)
        # print('selected cost', self.cost[proposed_cost > 0])
        # print('sum selected cost', np.sum(self.cost[proposed_cost > 0]))
        # print('delta', delta)
        # print((np.sum(proposed_cost) - np.sum(self.cost[proposed_cost > 0]) - delta))
        assert((np.sum(proposed_cost) - np.sum(self.cost[proposed_cost > 0]) - delta) < epsilon)
        return proposed_cost, delta

    def block2Coordinates(self):
        y_bound, x_bound = self.grid_size
        self.block_to_coordinates = np.zeros((self.num_blocks, 2), dtype=np.int32)

        for i in range(y_bound):
            for j in range(x_bound):
                current_cell = self.state[i, j]
                if current_cell != -1:
                    self.block_to_coordinates[current_cell] = (i, j)

    def annealWrapper(self):
        self.simulatedAnnealer()
        self.cost_var.set(np.sum(self.cost))
        self.plot()

    def simulatedAnnealer(self, init_temperature=1000, cooling_period=100, max_iter=2e6, early_stop_iter=2e4, beta=0.9):
        '''
        init_temperature: initial temperature
        cooling_period: num of iterations to decrease the temperature
        beta: cooling coefficient
        '''
        temp = init_temperature
        num_iter = 0
        early_stop = 0
        previous_best = np.iinfo(np.int32).max
        while True:
            for _ in range(cooling_period):
                # swap 2 random cooredinate
                # pair_to_swap = tuple(np.random.choice(self.num_blocks, 2, replace=False))
                # propose 2 coordinates instead of propose 2 blocks!
                pair_to_swap = self.proposeTwoPoint()

                proposed_cost, d_cost = self.evalDeltaCost(pair_to_swap)

                r = np.random.uniform()

                if r < np.exp(-d_cost / temp):
                    # swap
                    b1_coords, b2_coords = pair_to_swap
                    p1, p2 = tuple(b1_coords), tuple(b2_coords)
                    bid1, bid2 = self.state[p1], self.state[p2]
                    if bid1 == -1 and bid2 == -1:
                        print('sth is wrong!, not allowed to swap 2 empty cells')
                        assert(False)
                    elif bid2 == -1:
                        self.state[p1], self.state[p2] = -1, bid1
                        self.block_to_coordinates[bid1] = b2_coords
                    elif bid1 == -1:
                        self.state[p1], self.state[p2] = bid2, -1
                        self.block_to_coordinates[bid2] = b1_coords
                    else:
                        # bid1, bid2 = pair_to_swap
                        # b1_coords = tuple(self.block_to_coordinates[bid1])
                        # b2_coords = tuple(self.block_to_coordinates[bid2])
                        # print(bid1)
                        # print(type(self.block_to_coordinates[bid1]))
                        # update self.state and self.block_to_coordinates
                        # print('before state swap')
                        # print(self.state)
                        self.state[p1], self.state[p2] = bid2, bid1
                        # print('after state swap')
                        # print(self.state)
                        # print('before block 2 coordinates')
                        # print(self.block_to_coordinates)
                        self.block_to_coordinates[[bid1, bid2]] = self.block_to_coordinates[[bid2, bid1]]
                        # print('after block 2 coordinates')
                        # print(self.block_to_coordinates)
                    # print('before cost')
                    # print(self.cost)
                    self.cost[proposed_cost > 0] = proposed_cost[proposed_cost > 0]
                    # print('after cost')
                    # print(self.cost)
            temp *= beta
            num_iter += cooling_period
            if self.verbose:
                print('current iteration {}, current cost {}'.format(num_iter, np.sum(self.cost)))
            if np.sum(self.cost) < previous_best:
                previous_best = np.sum(self.cost)
                early_stop = 0
            else:
                early_stop += cooling_period
            if num_iter > max_iter:
                print('Reach maximum number of iterations, exit!')
                break
            if early_stop > early_stop_iter:
                print('Early stop!')
                break

    def proposeTwoPoint(self):
        '''
        Propose 2 unequal coordinates
        Also make sure not produce 2 empty cells
        '''
        y_bound, x_bound = self.grid_size
        while True:
            x = np.random.randint(0, x_bound, 2)
            y = np.random.randint(0, y_bound, 2)
            xy = np.stack([y, x]).T
            # xy = np.stack([x, y]).T
            p1, p2 = xy[0], xy[1]
            if (p1 - p2).any():  # look at 2 different points
                if self.state[tuple(p1)] != -1 or self.state[tuple(p2)] != -1:  # at least one point is a block
                    break
        return p1, p2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CPEN 513 Assignment 2: Placement')
    parser.add_argument('--infile', '-i', default='benchmarks/cm138a.txt', help='input file')  # yaml
    args = parser.parse_args()

    placer = Placer(args.infile, verbose=True)
    placer.simulatedAnnealer(init_temperature=1, cooling_period=100, early_stop_iter=2e3, beta=0.95)
    # placer.plot()
    placer.root.mainloop()
