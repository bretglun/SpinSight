from graphlib import TopologicalSorter
import numpy as np
from functools import partial


def equal(a, b):
    if a is b:
        return True
    
    if type(a) is not type(b):
        return False

    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)

    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(equal(a[k], b[k]) for k in a)

    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(equal(x, y) for x, y in zip(a, b))

    return a == b


class Node:
    
    def __init__(self, name, parents=None, func=None):
        self.name = name
        self.parents = parents or []
        self.func = func

        self._valid = False
        self._cache = None
        self.version = 0
        self.parent_versions = None
        self.children = []
        for parent in self.parents:
            parent.children.append(self)
    
    def invalidate(self):
        if not self._valid:
            return
        self._valid = False
        for child in self.children:
            child.invalidate()

    @property
    def value(self):
        if not self._valid:
            inputs = [parent.value for parent in self.parents]
            current_versions = tuple(p.version for p in self.parents)
            if current_versions != self.parent_versions:
                new = self.func(*inputs)
                if not equal(self._cache, new):
                    self.version += 1
                    self._cache = new
                self.parent_versions = current_versions if self.parents else None
            self._valid = True
        return self._cache


class Graph:

    def __init__(self, node_specs):
        ts = TopologicalSorter()
        for name, spec in node_specs.items():
            if 'parents' in spec:
                ts.add(name, *spec['parents'])
            else:
                ts.add(name)
        
        self.nodes = {}
        self.action_nodes = []
        for name in list(ts.static_order()):
            self.nodes[name] = self.make_node(name, node_specs[name])
            if node_specs[name].get('action', False):
                self.action_nodes.append(self.nodes[name])
        
        self.flush_actions()

    def flush_actions(self):
        for node in self.action_nodes:
            node.value

    def make_node(self, name, specs):
        parents = [self.nodes[parent] for parent in specs.get('parents', [])]
        func = specs.get('func', None)
        if func is None: # input node
            func = partial(getattr, specs.get('params'), name)
        return Node(name, parents=parents, func=func)


def print_dependency_chains(source, sink, chain=''):
    if not hasattr(sink, 'parents'):
        if sink.name == source.name:
            print(sink.name, chain)
        return
    name = sink.func.__name__
    if name.endswith('_func'):
        name = name[:-5]
    chain = f'-> {name} {chain}'
    for parent in sink.parents:
        print_dependency_chains(source, parent, chain)