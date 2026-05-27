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


class ParamNode:
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