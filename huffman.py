import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import datetime


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
    def __init__(self, value, id, state_dim, branch=16, hidden=64, learning_rate=1e-3, seed=1, max_seq_length=32):
        self.value = value
        self.id = id
        self.state_dim = state_dim
        self.branch = branch
        self.hidden_size = hidden
        self.lr = learning_rate
        self.seed, self.max_seq_length = seed, max_seq_length
        assert len(self.value) == len(self.id)
        self.tree = [Node(value=_) for _ in value]  # initialize the tree nodes
        self.root_child_count = 0
        for i, t in enumerate(self.tree):
            t.item_id = id[i]
        self.tree_copy = self.tree.copy()
        self.buildTree()
        self.codebook = self.getCode()
        self.input_state = tf.placeholder(dtype=tf.float32, shape=[1, self.max_seq_length, state_dim])
        self.input_state_length = tf.placeholder(dtype=tf.float32, shape=[None, ])
        self.input_reward = tf.placeholder(dtype=tf.float32, shape=[None, ])
        self.buildNetwork_v1()  # 构建网络
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

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
        self.root_child_count = len(self.tree)
        if len(self.tree) > 1:  # 如果最后根节点大于1，则需要新建个根节点
            v = sum([getValue(_) for _ in self.tree])
            new_node = Node(value=v, child=self.tree)
            for c in self.tree:
                c.father = new_node
            self.tree = new_node
        else:
            self.tree = self.tree[0]

    def getCode(self):
        # 得到所有的code
        codes = {}
        for node in self.tree_copy:
            code = ''
            node_id = node.item_id
            while node.father is not None:
                code = str(node.get_child_id()) + ' ' + code
                node = node.father
            codes[node_id] = code
        return codes

    def feature_extract(self, input_state, input_state_length):
        '''
        Create RNN feature extractor for recommendation systems.
        :return:
        '''
        with tf.variable_scope('feature_extract', reuse=False):
            basic_cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)
            _, states = tf.nn.dynamic_rnn(basic_cell, input_state, dtype=tf.float32,
                                          sequence_length=input_state_length)
        return states

    def mlp(self, id=None, softmax_activation=False):
        '''
        Create a multi-layer neural network as tree node.
        :param id: tree node id
        :param reuse: reuse for the networks
        :return: a multi-layer neural network with output dim equals to branch size.
        '''
        with tf.variable_scope('node_' + str(id), reuse=tf.AUTO_REUSE):
            state = self.feature_extract(self.input_state, self.input_state_length)
            l1 = slim.fully_connected(state, self.hidden_size)
            l2 = slim.fully_connected(l1, self.hidden_size)
            l3 = slim.fully_connected(l2, self.branch)
            if softmax_activation:
                outputs = tf.nn.softmax(l3)
            else:
                outputs = l3
        return outputs

    def buildNetwork_v1(self):
        # Build tree-structured neural networks for each node, without parameter sharing
        queue = []
        current_line = 0
        queue.append([current_line, self.tree])
        count = 0
        while len(queue) > 0:
            line, node = queue.pop(0)
            if line != current_line:  # for parameter sharing
                current_line = line
            node.network = self.mlp(id=str(count), softmax_activation=True)
            if len(node.child) != 0:
                for n in node.child:
                    queue.append([line + 1, n])
                    count += 1

    def buildNetwork_v2(self):
        # Build tree-structured neural networks for each node, layer parameter sharing
        # still have some bugs
        queue = []
        current_line = 0
        queue.append([current_line, self.tree])
        count = 0
        while len(queue) > 0:
            line, node = queue.pop(0)
            if line != current_line:  # for parameter sharing
                current_line = line
            node.network = self.mlp(id=str(count), softmax_activation=True)
            if len(node.child) != 0:
                for n in node.child:
                    queue.append([line + 1, n])
                    count += 1

    def learn(self, state, state_length, action, reward):
        # 根据输入的状态、动作和回报计算loss
        # 先根据action找到对应的path
        # 再根据path对路径进行遍历，计算总概率
        # 根据概率得出
        loss_list = []
        for i in range(len(action)):
            with tf.Session() as s:
                #tf.reset_default_graph()
                action_prob = self.getActionProb(action[i])
                loss = -self.input_reward * tf.log(action_prob + 1e-13)
                train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
                # self.sess.run(tf.variables_initializer(tf.report_uninitialized_variables()))
                s.run(tf.global_variables_initializer())
                #self.sess.run(tf.variables_initializer([loss]))
                _, l = s.run([train_op, loss],
                                     feed_dict={
                                         self.input_state: state[i].reshape([1, self.max_seq_length, self.state_dim]),
                                         self.input_state_length: [state_length[i], ],
                                         self.input_reward: [reward[i], ]})
            loss_list.append(l)
        return np.mean(loss_list)

    def getAction(self, state, state_length):
        # 根据输入的state和state_length得到一个动作的采样
        def softmax(x):
            """Compute the softmax in a numerically stable way."""
            x = x - np.max(x)
            exp_x = np.exp(x)
            softmax_x = exp_x / np.sum(exp_x)
            return softmax_x

        t_node = self.tree
        action_prob = self.sess.run(t_node.network,
                                    feed_dict={self.input_state: state, self.input_state_length: state_length})
        count = 0
        while len(t_node.child) != 0:
            if count == 0:  # root node
                np.random.seed(self.seed)
                c = np.random.choice(self.root_child_count, p=softmax(action_prob.flatten()[:self.root_child_count]))
            else:
                np.random.seed(self.seed)
                c = np.random.choice(self.branch, p=action_prob.flatten())
            t_node = t_node.child[c]
            action_prob = self.sess.run(t_node.network,
                                        feed_dict={self.input_state: state, self.input_state_length: state_length})
            count += 1
        return t_node.item_id

    def getActionProb(self, action):
        # 根据输入的state和state_length找出某个特定action的概率,是个tensor
        path = list(map(int, self.codebook[action].split()))
        node = self.tree
        action_prob = 1
        for i in range(len(path)):
            t = node.network[:, path[i]]
            action_prob *= node.network[:, path[i]]
            node = node.child[path[i]]
        return action_prob

    def state_padding(self, input_state, input_state_length):
        if input_state_length > self.max_seq_length:
            input_state = input_state[-self.max_seq_length:]
            input_state_length = self.max_seq_length
        input_state = np.array(input_state).reshape([input_state_length, self.state_dim])
        if input_state_length < self.max_seq_length:
            # padding the zero matrix.
            padding_mat = np.zeros([self.max_seq_length - input_state_length, self.state_dim])
            input_state = np.vstack((input_state, padding_mat))
        return input_state


id = np.arange(100)
np.random.seed(1)
frequency = np.arange(100)
h_tree = HuffmanTree(value=frequency, id=id, branch=3, state_dim=10)

state = np.random.rand(1, 32, 10)
state_length = 10
action = h_tree.getAction(state, [state_length])
action_prob = h_tree.getActionProb(1)

for i in range(100):
    start = datetime.datetime.now()
    loss = h_tree.learn(state, [state_length], [1], [10])
    end = datetime.datetime.now()
    print('Step {}\n loss:{} time:{}'.format(i,loss,(end-start).seconds))
print('Training time:{}'.format((end - start).seconds))
pass
