import functools

from pesnet.systems.system import Atom, Molecule

@functools.lru_cache
def read_xyz(file_name):
    with open(file_name) as inp:
        lines = inp.readlines()

    i = 0
    configurations = {}
    while i <= len(lines):
        try:
            try:
                n = int(lines[i])
            except ValueError:
                i += 1
                n = int(lines[i])
            name = lines[i+1].strip('\n \t')
        except IndexError:
            break
        i += 2
        configs = lines[i:i+n]
        i += n
        configurations[name] = []
        for c in configs:
            splits = [s for s in c.strip('\n \t').split(' ') if len(s) > 0]
            element = splits[0]
            coords = [float(x) for x in splits[1:4]]
            configurations[name].append(Atom(element, coords, units='angstrom'))
    
    for k in configurations.keys():
        configurations[k] = Molecule(configurations[k])

    return configurations
