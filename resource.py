from datetime import datetime
import numpy as np

slots = [0, 3, 6, 9, 12, 15, 18, 21]


class Observatory:
    def __init__(self, name):
        self.path = 'data/{n}.dat'.format(n=name)
        self.metadata = dict()

        file = open(self.path)
        cs = file.readline().strip('# \n').split(',')
        self.metadata[cs[0]] = cs[1]
        cs = file.readline().strip('# \n').split(',')
        self.metadata[cs[0]] = cs[1]
        cs = file.readline().strip('# \n').split(',')
        self.metadata[cs[0]] = float(cs[1])
        cs = file.readline().strip('# \n').split(',')
        self.metadata[cs[0]] = float(cs[1])
        cs = file.readline().strip('# \n').split(',')
        self.metadata[cs[0]] = float(cs[1])
        cs = file.readline().strip('# \n').split(',')
        self.metadata[cs[0]] = datetime.strptime(cs[1], '%Y-%m-%d %H:%M %j')
        cs = file.readline().strip('# \n').split(',')
        self.metadata[cs[0]] = datetime.strptime(cs[1], '%Y-%m-%d %H:%M %j')
        cs = file.readline().strip('# \n').split(',')
        self.metadata[cs[0]] = int(cs[1])
        print('Header Initialization done')
        file.close()

        self.data = np.zeros((self.metadata['Days'], 8, 180))

    def parse(self):
        file = open(self.path, 'r')
        for line in file.readlines():
            if line.startswith('#'):
                continue
            cs = line.strip('\n').split(',')
            self.data[int(cs[0])][int(cs[1])][int(cs[2])] = float(cs[3])

        print('Data Initialization complete')
        file.close()
