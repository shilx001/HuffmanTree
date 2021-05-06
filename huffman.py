import tensorflow as tf
import numpy as np


class Node:
    def __init__(self, value=0, child=[], network=None):
        self.value = value
        self.child = child
        self.father = None
        self.item_id = None
        self.network = network

    def get_child(self):
        return self.child

    def get_child_id(self):
        # 找出是第几个孩子，如果是其儿子，则返回子节点,否则返回-1
        c = self.father.get_child()
        for i, t in enumerate(c):
            if t == self:
                return i
        else:
            return -1


class HuffmanTree:
    def __init__(self, value, id, branch=16):
        self.value = value
        self.id = id
        self.branch = branch
        assert len(self.value) == len(self.id)
        self.tree = [Node(value=_) for _ in value]  # initialize the tree nodes
        for i, t in enumerate(self.tree):
            t.item_id = id[i]
        self.tree_copy = self.tree.copy()

    def buildTree(self):
        # Organize the tree nodes to huffman tree.
        def getValue(n):
            return n.value

        while len(self.tree) >= self.branch:
            self.tree.sort(key=lambda n: n.value)
            child = self.tree[:self.branch]
            v = sum([getValue(_) for _ in child])
            new_node = Node(value=v, child=child)
            for c in child:
                c.father = new_node
            self.tree = self.tree[self.branch:]
            self.tree.append(new_node)

    def getCode(self):
        #得到所有的code
        codes = {}
        for node in self.tree_copy:
            code = ''
            node_id = node.item_id
            while node.father is not None:
                code = str(node.get_child_id()) + ' ' + code
                node = node.father
            codes[node_id] = code
        return codes

    def getProbability(self):
        # 根据输入的动作编码
        pass

id = np.arange(10)
np.random.seed(1)
frequency = np.random.random_integers(0,100,size=[10,])
h_tree = HuffmanTree(value=frequency,id=id, branch=2)
h_tree.buildTree()
code=h_tree.getCode()