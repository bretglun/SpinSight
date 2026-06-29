from graphlib import TopologicalSorter
import numpy as np
from functools import partial
import inspect


class Graph:

    node_specs = {}

    @classmethod
    def node(cls, action=False):
        def decorator(func):
            func_params = [p.name for p in inspect.signature(func).parameters.values()]
            cls.node_specs[func.__name__] = {
                'func': func,
                'parents': [p for p in func_params if p != 'self'],
                'action_precedence': action # order for action node to be flushed
            }
            return func
        return decorator

    def __init__(self, simulator):
        self.simulator = simulator

        specs = self.build_node_specs()
        self.nodes, self.action_nodes = self.build_nodes(specs)
        
        self.processing = False

        self.flush_actions()
    
    def build_node_specs(self):
        # get node specs from decorators
        specs = dict(type(self).node_specs)
        # special node to track which input node was trigger
        specs['trigger_node'] = {'func': lambda: 'None'}
        # special simulator node
        specs['simulator'] = {'func': lambda: self.simulator}
        # add specs for remaining simulator param nodes
        specs.update(self.simulator.get_input_node_specs())
        return specs
    
    def build_nodes(self, specs):
        nodes, action_nodes = {}, {}
        for name in topological_order(specs):
            nodes[name] = Node(name, specs[name].get('func', None))
            for parent in specs[name].get('parents', []):
                nodes[name].attach(nodes[parent])
            if specs[name].get('action_precedence', False):
                precedence = specs[name].get('action_precedence')
                if precedence not in action_nodes:
                    action_nodes[precedence] = []
                action_nodes[precedence].append(nodes[name])
        return nodes, action_nodes
    
    def on_change(self, node, event):
        if not self.processing:
            self.nodes['trigger_node'].func = lambda: node.name
            self.nodes['trigger_node'].invalidate()
        was_processing = self.processing
        self.processing = True
        node.invalidate()
        self.flush_actions()
        self.processing = was_processing
    
    def flush_actions(self):
        for precedence in sorted(self.action_nodes):
            for node in self.action_nodes[precedence]:
                node.value

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
            if not self.parents or current_versions != self.parent_versions:
                self.parent_versions = current_versions
                self.recompute(inputs)
            self._valid = True
        return self._cache
    
    def recompute(self, inputs):
        new = self.func(*inputs)
        if not equal(self._cache, new):
            self.version += 1
            self._cache = new


def print_dependency_chains(source, sink, chain=''):
    if sink.name == source.name:
        print(sink.name, chain)
        return
    elif not sink.parents:
        return
    chain = f'-> {sink.name} {chain}'
    for parent in sink.parents:
        print_dependency_chains(source, parent, chain)


def topological_order(specs):
    ts = TopologicalSorter()
    for name, spec in specs.items():
        ts.add(name, *spec.get('parents', []))
    return ts.static_order()


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