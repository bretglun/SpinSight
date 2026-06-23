from graphlib import TopologicalSorter
import numpy as np
from functools import partial
import inspect


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
        self.func = func
        
        self.parents = []
        for parent in parents or []:
            self.attach(parent)
        
        self.children = []

        self._valid = False # parent nodes may have changed
        self._cache = None
        self.version = 0
        self.parent_versions = ()
    
    def attach(self, parent):
        self.parents.append(parent)
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
            if current_versions != self.parent_versions or not self.parents:
                self.recompute(inputs)
                self.parent_versions = current_versions
            self._valid = True
        return self._cache
    
    def recompute(self, inputs):
        new = self.func(*inputs)
        if not equal(self._cache, new):
            self.version += 1
            self._cache = new


class Graph:

    node_specs = {}
    
    @classmethod
    def node(cls, action=False):
        def decorator(func):
            func_params = [p.name for p in inspect.signature(func).parameters.values()]
            cls.node_specs[func.__name__] = {
                'func': func,
                'parents': [p for p in func_params if p != 'self'],
                'simulator': 'self' in func_params, #TODO: nicer!
                'action': action
            }
            return func
        return decorator

    def __init__(self, simulator):

        self.simulator = simulator
        
        self.node_specs.update({par: {'params': True} for par in simulator.param if par != 'name' and par not in self.node_specs})

        ts = TopologicalSorter()
        for name, spec in self.node_specs.items():
            if 'parents' in spec:
                ts.add(name, *spec['parents'])
            else:
                ts.add(name)
        
        self.nodes = {}
        self.action_nodes = []
        for name in list(ts.static_order()):
            self.nodes[name] = self.make_node(name, self.node_specs[name])
            if self.node_specs[name].get('action', False):
                self.action_nodes.append(self.nodes[name])
        
        self.flush_actions()

    def flush_actions(self):
        for node in self.action_nodes:
            node.value

    def make_node(self, name, specs):
        parents = [self.nodes[parent] for parent in specs.get('parents', [])]
        func = specs.get('func', None)
        if func is None: # input node
            func = partial(getattr, self.simulator, name)
        elif specs.get('simulator', False): # simulator node (TODO: nicer!)
            func = partial(func, self.simulator)
        return Node(name, parents=parents, func=func)


def print_dependency_chains(source, sink, chain=''):
    if not hasattr(sink, 'parents'):
        if sink.name == source.name:
            print(sink.name, chain)
        return
    chain = f'-> {sink.func.__name__} {chain}'
    for parent in sink.parents:
        print_dependency_chains(source, parent, chain)