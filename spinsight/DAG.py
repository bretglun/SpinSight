from graphlib import TopologicalSorter


class GraphScheduler:

    def __init__(self):
        self.pending_actions = []
        self.invalidating = False

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


scheduler = GraphScheduler()


class InputParamNode:
    def __init__(self, params, name):
        self.params = params
        self.name = name
        self.children = []
        params.param.watch(self._on_change, name)

    @property
    def value(self):
        return getattr(self.params, self.name)

    def _on_change(self, event):
        scheduler.begin_invalidation()
        for child in self.children:
            child.invalidate()
        scheduler.end_invalidation()


class ComputeNode:
    def __init__(self, func, parents=[]):
        self.func = func
        self.parents = parents
        self._valid = False
        self._cache = None
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
            self._cache = self.func(*inputs)
            self._valid = True
        return self._cache
    

class ActionNode:
    def __init__(self, func, parents):
        self.func = func
        self.parents = parents
        self._queued = False
        for parent in self.parents:
            parent.children.append(self)

    def execute(self):
        self.func(*[parent.value for parent in self.parents])
        self._queued = False
    
    def invalidate(self):
        if not self._queued:
            scheduler.queue_action(self)
            self._queued = True


class OutputParamNode(ComputeNode):
    def __init__(self, params, name, func, parents):
        super().__init__(func, parents)
        self.params = params
        self.name = name
        self._queued = False

    @property
    def value(self):
        value = super().value
        current = getattr(self.params, self.name)
        if current != value:
            setattr(self.params, self.name, value)
        return value

    def execute(self):
        self.value
        self._queued = False
    
    def invalidate(self):
        super().invalidate()
        if not self._queued:
            scheduler.queue_action(self)
            self._queued = True


def make_node(name, specs, graph):
    if 'parents' not in specs:
        if 'params' not in specs:
            raise ValueError(f'A node must have "parents" or "params" (for input nodes). Node "{name}" does not.')
        return InputParamNode(specs['params'], name)
    if 'func' not in specs:
        raise ValueError(f'A node with "parents" must also have "func". Node "{name}" does not.')
    parents = [graph[parent] for parent in specs['parents']]
    if specs.get('action', False):
        return ActionNode(specs['func'], parents)
    elif 'params' not in specs:
        return ComputeNode(specs['func'], parents)
    return OutputParamNode(specs['params'], name, specs['func'], parents)


def initialize_graph(graph):
    for node in graph.values():
        if isinstance(node, ActionNode) or isinstance(node, OutputParamNode):
            node.execute()


def build_graph(node_specs):
    ts = TopologicalSorter()
    for name, spec in node_specs.items():
        if 'parents' in spec:
            ts.add(name, *spec['parents'])
        else:
            ts.add(name)
    graph = {}
    for name in list(ts.static_order()):
        graph[name] = make_node(name, node_specs[name], graph)
    initialize_graph(graph)
    return graph


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