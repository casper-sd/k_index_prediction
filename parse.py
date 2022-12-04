import os
import re
from os.path import exists
from datetime import datetime, timedelta
from math import sqrt, floor
from resource import slots


class Observatory:
    headers = ['Station Name',
               'IAGA CODE',
               'Geodetic Latitude',
               'Geodetic Longitude',
               'Elevation',
               'Reported'
               ]

    def __init__(self, dir_path):
        if not exists(dir_path):
            print("No such directory")
            return
        self.metadata = dict()
        self.data = dict()
        self.dir_path = dir_path

    def get_header(self, f):
        metadata = dict()
        for i in range(12):
            h = next(f)
            for head in Observatory.headers:
                if h.startswith(' ' + head):
                    v = h[len(head)+1:].strip(' |\n')
                    if len(v) != 0:
                        metadata[head] = v

        if not self.metadata:
            self.metadata = metadata
        else:
            assert metadata['IAGA CODE'] == self.metadata['IAGA CODE'] and \
                   metadata['Station Name'] == self.metadata['Station Name']
        for h in Observatory.headers:
            if h in metadata.keys() and h not in self.metadata.keys():
                self.metadata[h] = metadata[h]
        self.metadata['Reported'] = metadata['Reported']

    def get_data(self, f):
        for line in f.readlines():
            if line.endswith('|\n'):
                continue
            line_comps = re.split(' +', line.rstrip('\n'))
            dt = datetime.strptime(' '.join(line_comps[:3]), '%Y-%m-%d %H:%M:%S.%f %j')
            if 'X' not in self.metadata['Reported'] or 'Y' not in self.metadata['Reported'] or \
                    'Z' not in self.metadata['Reported']:
                assert 'H' in self.metadata['Reported'] and 'Z' in self.metadata['Reported']
                h = str(self.metadata['Reported']).find('H') + 3
                H = float(line_comps[h])
                self.data[dt] = H
            else:
                x = str(self.metadata['Reported']).find('X') + 3
                y = str(self.metadata['Reported']).find('Y') + 3
                X = float(line_comps[x])
                Y = float(line_comps[y])
                self.data[dt] = sqrt(X**2 + Y**2)

    def parse(self):
        f_count = 0
        for file in os.listdir(self.dir_path):
            f = open(self.dir_path + '/' + file)
            self.get_header(f)
            f.seek(0, 0)
            self.get_data(f)
            f.close()
            f_count += 1

        self.metadata['Start Time'] = min(self.data.keys())
        self.metadata['End Time'] = max(self.data.keys()) + timedelta(minutes=1)
        self.metadata['Geodetic Latitude'] = float(self.metadata['Geodetic Latitude'])
        self.metadata['Geodetic Longitude'] = float(self.metadata['Geodetic Longitude'])
        self.metadata['Elevation'] = float(self.metadata['Elevation'])
        self.metadata['Days'] = f_count
        print('Parsing complete')


station_name = 'BOU'
obs = Observatory(dir_path='D:/intermagnet/' + station_name)
obs.parse()

file = open('data/{sn}.dat'.format(sn=station_name), 'w')
for k, v in obs.metadata.items():
    if k == 'Reported':
        continue
    if type(v) == datetime:
        file.write('# ' + k + ',' + v.strftime('%Y-%m-%d %H:%M %j') + '\n')
    else:
        file.write('# ' + k + ',' + str(v) + '\n')

dt = obs.metadata['Start Time']
st = obs.metadata['Start Time']
while dt < obs.metadata['End Time']:
    slot = floor(dt.hour / 3)
    day = (dt.date() - st.date()).days
    inst = (dt.hour - slots[slot])*60 + dt.minute
    file.write(str(day) + ',' + str(slot) + ',' + str(inst) + ',' + str(obs.data[dt]) + '\n')
    dt += timedelta(minutes=1)

file.close()
