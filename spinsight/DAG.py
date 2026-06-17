from graphlib import TopologicalSorter
import numpy as np


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


class InputParamNode:
    def __init__(self, params, name, graph):
        self.params = params
        self.name = name
        self.graph = graph
        self.version = 0
        self.children = []
        params.param.watch(self._on_change, name)

    @property
    def value(self):
        return getattr(self.params, self.name)

    def _on_change(self, event):
        self.version += 1
        self.graph.begin_invalidation()
        for child in self.children:
            child.invalidate()
        self.graph.end_invalidation()


class ComputeNode:
    def __init__(self, func, parents=[]):
        self.func = func
        self.parents = parents
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
                old = self._cache
                new = self.func(*inputs)
                self.parent_versions = current_versions
                if not equal(old, new):
                    self._cache = new
                    self.version += 1
            self._cache = self.func(*inputs)
            self._valid = True
        return self._cache


class ActionNode:
    def __init__(self, func, parents, graph):
        self.func = func
        self.parents = parents
        self.graph = graph
        self.parent_versions = None
        for parent in self.parents:
            parent.children.append(self)

    def execute(self):
        inputs = [parent.value for parent in self.parents]
        current_versions = tuple(p.version for p in self.parents)
        if current_versions != self.parent_versions:
            self.func(*inputs)
            self.parent_versions = current_versions
    
    def invalidate(self):
        self.graph.queue_action(self)


class OutputParamNode(ComputeNode):
    def __init__(self, params, name, func, parents, graph):
        super().__init__(func, parents)
        self.params = params
        self.name = name
        self.graph = graph

    @property
    def value(self):
        value = super().value
        current = getattr(self.params, self.name)
        if current != value:
            setattr(self.params, self.name, value)
        return value

    def execute(self):
        self.value
    
    def invalidate(self):
        super().invalidate()
        self.graph.queue_action(self)


class Graph:

    def __init__(self, node_specs):
        ts = TopologicalSorter()
        for name, spec in node_specs.items():
            if 'parents' in spec:
                ts.add(name, *spec['parents'])
            else:
                ts.add(name)
        self.nodes = {}
        for name in list(ts.static_order()):
            self.nodes[name] = self.make_node(name, node_specs[name])
        
        self.pending_actions = []
        self.invalidating = False

        for node in self.nodes.values():
            if isinstance(node, ActionNode) or isinstance(node, OutputParamNode):
                node.execute()
    
    def begin_invalidation(self):
        self.invalidating = True

    def end_invalidation(self):
        self.invalidating = False
        self.flush_actions()

    def queue_action(self, action_node):
        if action_node not in self.pending_actions:
            self.pending_actions.append(action_node)

    def flush_actions(self):
        while self.pending_actions:
            action_node = self.pending_actions.pop(0)
            action_node.execute()

    def make_node(self, name, specs):
        if 'parents' not in specs:
            if 'params' not in specs:
                raise ValueError(f'A node must have "parents" or "params" (for input nodes). Node "{name}" does not.')
            return InputParamNode(specs['params'], name, self)
        if 'func' not in specs:
            raise ValueError(f'A node with "parents" must also have "func". Node "{name}" does not.')
        parents = [self.nodes[parent] for parent in specs['parents']]
        if specs.get('action', False):
            return ActionNode(specs['func'], parents, self)
        elif 'params' not in specs:
            return ComputeNode(specs['func'], parents)
        return OutputParamNode(specs['params'], name, specs['func'], parents, self)


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