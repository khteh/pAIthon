class Person():
    def __init__(self, id, parent, movie):
        self.id = id
        self.parent = parent
        self.movie = movie

class StackFrontier():
    def __init__(self):
        self.frontier = []

    def add(self, node):
        self.frontier.append(node)

    def contains_id(self, id):
        return any(node.id == id for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def pop(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[-1]
            self.frontier = self.frontier[:-1]
            return node

class QueueFrontier(StackFrontier):
    def isEmpty(self):
        return self.empty()
    
    def pop(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[0]
            self.frontier = self.frontier[1:]
            return node
