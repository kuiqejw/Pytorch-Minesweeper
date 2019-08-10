import sys
sys.setrecursionlimit(1000000)
import random
import numpy as np

class Minesweeper(object):
    def __init__(self, shape):
        super(Minesweeper, self).__init__()
        self.shape = shape
        self.status = 0

        if self.shape == 'easy':
            self.width = 8
            self.height = 8
            self.n_mines = 5
        elif self.shape == 'middle':
            self.width = 16
            self.height = 16
            self.n_mines = 40
        elif self.shape == 'hard':
            self.width = 30
            self.height = 16
            self.n_mines = 99
        self.uncleared_blocks = self.width * self.height - self.n_mines

        self.map = np.zeros((self.height, self.width)) # [[0 for i in range(self.width)] for j in range(self.height)]
        self.mines = np.zeros((self.height, self.width)) # [[0 for i in range(self.width)] for j in range(self.height)]
        self.mask = np.zeros((self.height, self.width)) # [[-1 for i in range(self.width)] for j in range(self.height)]
        # self.mask.fill(-1)
        
        for index in random.sample(range(1, self.width * self.height), self.n_mines):
            self.map[index // self.width][index % self.width] = 1

        for i in range(self.height):
            for j in range(self.width):
                self.mines[i][j] = self.get_mine_num(i, j)
                if self.map[i][j] == 1:
                    self.mines[i][j] = 9

    def action(self, a):
        x = a // self.width
        y = a % self.width
        try:
            self.mask[x][y] = self.mines[x][y]
        except IndexError:
            print(x, y)

        if self.map[x][y] == 1:
            self.status = -1
            print('failed!')
        else:
            if self.mines[x][y] == -1:
                self.clear_empty_blocks(x, y)
            self.uncleared_blocks = self.width * self.height - self.n_mines - (self.mask != 0).sum()
            if self.uncleared_blocks == 0:
                self.status = 1
                print('win!')

    def clear_empty_blocks(self, i, j):
        self.mask[i][j] = self.mines[i][j]
        if self.mines[i][j] != -1:
            return
        else:
            neighbours = [(i-1, j-1), (i-1, j), (i-1, j+1),
                          (i, j-1), (i, j+1),
                          (i+1, j-1), (i+1, j), (i+1, j+1)]
            for n in neighbours:
                if n[0] == -1 or n[1] == -1 or n[0] >= self.height or n[1] >= self.width or self.mask[n[0]][n[1]] != 0:
                    continue
                self.clear_empty_blocks(n[0], n[1])

    def get_score(self):
        sum = 0
        for i in self.mask:
            for j in i:
                if j != -1:
                    sum += 1
        return sum

    def get_state(self):
        return self.mask.copy()

    def get_status(self):
        return self.status

    def show(self):
        print('======MAP======')
        for i in range(self.height):
          print(self.map[i])
        # print('======MASK======')
        # for i in range(self.height):
        #     print(self.mask[i])
        # print('======MINE======')
        # for i in range(self.height):
        #    print(self.mines[i])

    def get_mine_num(self, i, j):
        neighbours = [(i-1, j-1), (i-1, j), (i-1, j+1),
                      (i, j-1), (i, j), (i, j+1),
                      (i+1, j-1), (i+1, j), (i+1, j+1)]
        mine_num = 0
        for n in neighbours:
            if n[0] == -1 or n[1] == -1 or n[0] >= self.height or n[1] >= self.width:
                continue
            if self.map[n[0]][n[1]] == 1:
                mine_num += 1
        if mine_num == 0:
            mine_num = -1
        return mine_num

game = Minesweeper('easy')
game.show()
while (game.get_status() == 0):
   a = input('input a: ')
   game.action(int(a))
   game.show()
