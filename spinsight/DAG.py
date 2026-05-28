class InputParamNode:
    def __init__(self, params, name):
        self.params = params
        self.name = name

        self.children = []

        params.param.watch(
            self._on_change,
            name
        )

    @property
    def value(self):
        return getattr(self.params, self.name)

    def _on_change(self, event):
        for child in self.children:
            child.invalidate()


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

        for parent in self.parents:
            parent.children.append(self)

    def execute(self):
        self.func(*[parent.value for parent in self.parents])
    
    def invalidate(self):
        self.execute()


class OutputParamNode(ComputeNode):
    def __init__(self, params, name, func, parents):
        super().__init__(func, parents)

        self.params = params
        self.name = name

    @property
    def value(self):

        value = super().value

        current = getattr(self.params, self.name)

        if current != value:
            setattr(self.params, self.name, value)

        return value