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


def topological_order(specs):
    ts = TopologicalSorter()
    for name, spec in specs.items():
        ts.add(name, *spec.get('parents', []))
    return ts.static_order()


class Node:
    
    def __init__(self, name, func=None):
        self.name = name
        self.func = func
        
        self.parents = []
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
    def node(cls, action=False, simulator=False):
        def decorator(func):
            func_params = [p.name for p in inspect.signature(func).parameters.values()]
            cls.node_specs[func.__name__] = {
                'func': func,
                'parents': [p for p in func_params if p != 'self'],
                'simulator': simulator, # is func a method of MRIsimulator?
                'action': action # action node to be flushed?
            }
            return func
        return decorator

    def __init__(self, simulator):
        self.simulator = simulator

        specs = self.build_node_specs()
        self.nodes, self.action_nodes = self.build_nodes(specs)

        self.flush_actions()
        self.add_input_watchers()
    
    def build_node_specs(self):
        # get node specs from decorators
        specs = dict(type(self).node_specs)
        # add specs for input nodes
        specs.update({par: {'params': True} for par in self.simulator.param if par != 'name' and par not in specs})
        return specs
    
    def build_nodes(self, specs):
        nodes, action_nodes = {}, []
        for name in topological_order(specs):
            nodes[name] = self.make_node(name, specs[name].get('func', None), specs[name].get('simulator', False))
            for parent in specs[name].get('parents', []):
                nodes[name].attach(nodes[parent])
            if specs[name].get('action', False):
                action_nodes.append(nodes[name])
        
        return nodes, action_nodes

    def make_node(self, name, func, simulator):
        if func is None: # input node
            func = partial(getattr, self.simulator, name)
        elif simulator:
            func = partial(func, self.simulator)
        return Node(name, func=func)
    
    def add_input_watchers(self):
        for node in self.input_nodes():
            def on_change(node, graph, event):
                node.invalidate()
                graph.flush_actions()
            self.simulator.param.watch(partial(on_change, node, self), node.name)
    
    def input_nodes(self):
        return [node for node in self.nodes.values() if (node.name in self.simulator.param) and (node.name not in self.simulator.derived_params)]
    
    def flush_actions(self):
        for node in self.action_nodes:
            node.value


def print_dependency_chains(source, sink, chain=''):
    if not hasattr(sink, 'parents'):
        if sink.name == source.name:
            print(sink.name, chain)
        return
    chain = f'-> {sink.func.__name__} {chain}'
    for parent in sink.parents:
        print_dependency_chains(source, parent, chain)