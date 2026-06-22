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
    # TODO: try to make composite of classes with a single responsibility
    def __init__(self, name, parents=None, func=None, graph=None, params=None):
        self.name = name
        self.parents = parents or []
        self.func = func
        self.graph = graph
        self.params = params

        self._valid = False
        self._cache = None
        self.version = 0
        self.parent_versions = None
        self.children = []
        for parent in self.parents:
            parent.children.append(self)

        if not parents and params is not None:
            params.param.watch(self._on_change, name)
        
        if parents and graph is not None:
            graph.queue_action(self)

    def _on_change(self, event):
        self.invalidate()
        self.graph.flush_actions()
    
    def invalidate(self):
        if not self._valid:
            return
        self._valid = False
        for child in self.children:
            child.invalidate()
        if self.graph is not None: # TODO: should add: and self.params is not None:
            self.graph.queue_action(self)

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
                    if self.params and new != getattr(self.params, self.name):
                        setattr(self.params, self.name, new)
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
        
        self.pending_actions = []
        
        self.nodes = {}
        for name in list(ts.static_order()):
            self.nodes[name] = self.make_node(name, node_specs[name])
        
        self.flush_actions()

    def queue_action(self, action_node):
        if action_node not in self.pending_actions:
            self.pending_actions.append(action_node)

    def flush_actions(self):
        while self.pending_actions:
            action_node = self.pending_actions.pop(0)
            action_node.value

    def make_node(self, name, specs):
        parents = [self.nodes[parent] for parent in specs.get('parents', [])]
        func = specs.get('func', None)
        action = specs.get('action', False)
        params = specs.get('params', None)
        graph = self if action or params is not None else None
        if func is None: # input node
            func = partial(getattr, params, name)
        return Node(name, parents=parents, func=func, graph=graph, params=params)


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