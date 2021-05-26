## MENU
- [MENU](#menu)
- [Lecture 1](#lecture-1)
  - [What is an Intelligent agents:](#what-is-an-intelligent-agents)
- [Lecture 2 Problem Formulation, State Space and Search Tree](#lecture-2-problem-formulation-state-space-and-search-tree)
  - [Problem Formulation](#problem-formulation)
  - [Problem Components:](#problem-components)
  - [Tree](#tree)
- [Lecture 3 Blind Searches](#lecture-3-blind-searches)
  - [Strategies that evaluated along the dimensions:](#strategies-that-evaluated-along-the-dimensions)
  - [Breadth First Search](#breadth-first-search)
  - [Depth First Search](#depth-first-search)
  - [Uniform Cost Search](#uniform-cost-search)
- [Lecture 4 Heuristic Searches 启发式搜索](#lecture-4-heuristic-searches-启发式搜索)
  - [Heuristic search(informed search):](#heuristic-searchinformed-search)
  - [Greedy Search](#greedy-search)
    - [Compared with UCS](#compared-with-ucs)
  - [A* Algorithm](#a-algorithm)
    - [How to estimate h:](#how-to-estimate-h)
    - [Effective Branching Factor](#effective-branching-factor)
- [Lecture 5 Game Playing](#lecture-5-game-playing)
  - [Game Playing - MINIMAX](#game-playing---minimax)
  - [Zero-Sum Games](#zero-sum-games)
  - [Components of Game Search](#components-of-game-search)
    - [Game Playing - MINIMAX](#game-playing---minimax-1)
  - [Alpha-Beta Pruning(修剪)](#alpha-beta-pruning修剪)
  - [Game classification](#game-classification)
  - [Generative Adversarial Network GAN](#generative-adversarial-network-gan)
    - [GANs:](#gans)
    - [Two-Player game:](#two-player-game)
- [Lecture 6 Machin Learning & Data Mining](#lecture-6-machin-learning--data-mining)
  - [Supervised learning](#supervised-learning)
  - [Unsupervised learning](#unsupervised-learning)
  - [Data Mining:](#data-mining)
    - [Data Mining Tasks:](#data-mining-tasks)
  - [CLassfication Methods:](#classfication-methods)
    - [Classification: K-Nearest Neigbor (KNN)](#classification-k-nearest-neigbor-knn)
  - [Clustering](#clustering)
    - [Clustering Methods](#clustering-methods)
- [Lecture 7 Nerual Networks](#lecture-7-nerual-networks)
  - [The first Neural Networks](#the-first-neural-networks)
  - [Activation Function:](#activation-function)
  - [Neural Network Architectures](#neural-network-architectures)
    - [Three different classes of network architectures](#three-different-classes-of-network-architectures)
  - [Perceptron 感知器](#perceptron-感知器)
  - [Linear Separability](#linear-separability)
    - [Epoch:](#epoch)
  - [The learning process: 学习进程](#the-learning-process-学习进程)
- [Lecture 8 Probabilistic Reasoning and Bayes' Theorem](#lecture-8-probabilistic-reasoning-and-bayes-theorem)
  - [Basic Concepts:](#basic-concepts)
  - [Probability Theory](#probability-theory)
  - [Bayes' Theorem](#bayes-theorem)
  - [Random Variables:](#random-variables)
  - [Joint Probability Distribution](#joint-probability-distribution)
## Lecture 1

### What is an Intelligent agents:
+ An intelligent agent is a system that ``perceives its environment`` and ``takes actions`` that `maximize its chances of success`
  + AI 感知环境并且采取行动，来最大化成功的机会
+ Preceives its environment:
  + Observation and understanding of the environment
  + Observation => Facts (machine learning, pattern recognition)
  + Facts => knowledge (data mining, knowledge representation adn reasoning, searching)
+ Take actions
  + making decisions

## Lecture 2 Problem Formulation, State Space and Search Tree

### Problem Formulation 
+ 问题描述
+ Problem formulation is the process of deciding what actions and states to consider, given a goal.
  + 在给定一个目标的情况下决定采取什么措施

### Problem Components:
+ ``Initial State``: 问题的起始状态
+ ``Actions(Operators)``: 
  + An action or a set of actions that moves the problem from one state to another
  + ``Neighbourhood``: 一个状态附近的所有执行一个合法action可以到达的状态都是它的邻居, the actions can be recognized the ``successor function``后续功能 
+ ``Goal Test``:
  + A test applied to a state which returns true if we have reached a state that solves the problem 检查是否到了目标状态点
+ ``Path Cost``: 走过一段路径到终点花费的总时间

### Tree
+ ``Average branching factor``: average number of branches of the nodes in the tree 节点的平均分支数

## Lecture 3 Blind Searches
### Strategies that evaluated along the dimensions:
+ ``Completeness``: 是否完成了搜索
+ ``time complexity``
+ ``space complexity``: maximum number of nodes in memory 内存中的最大的node数量
+ ``optimality``: 是否总能找到最优解

### Breadth First Search
+ Using Queuing function: 把节点添加到队列末端 (FIFO) 先进先出
```
node = Remove-Front(frontier)
if Goal-Test[p] on State(node) succeeds:
    then return node
frontier = QUEUING-FN(frontier, Expand(node,Action[p]))
```
+ 搜索树中的三种类型节点
  + frontier nodes
    + 被搜索过的但还没有被处理过：没有检索过他们的子节点有没有目标
  + visited nodes(closed nodes)
    + 被搜索过了，他们的子节点也都被探索过了
  + Undiscovered nodes:
    + 还没有过搜索到的节点

+ 评估：
  + ``Optimal`` (YES)
  + ``Complete`` (YES)
  + ``Space Complexity`` O(b^d) branch factor ** depth i.e. number of nodes in the tree
  + ``Time Complexity`` O(b^d) i.e. total number of nodes in the tree
    + b: the maximum branching factor
    + d: the depth of the search tree
  + ``Exponential Growth``: Combinatorial explostion 组合爆炸，指数增长

### Depth First Search
+ adds node to the front of the queue (LIFO) 使用一个后进先出的队列

+ 评估
  + ``Space complexity``:
    + 会存储从一个到根节点到子叶的路径和还没有拓展的邻居节点
    + O(bm)
  + ``Time complexity``
    + O(b^m)
    + b: branching factor
    + m: maximum depth
  + ``Completeness``: 如果遇到一个无限的分支，那么有可能无法结束
  + ``Optimality``: 不一定是最优的

### Uniform Cost Search
+ always remove the smallest cost node first
  + 总是先探索cost最低的节点
    + 将队列根据cost进行排列

## Lecture 4 Heuristic Searches 启发式搜索
### Heuristic search(informed search):
+ using knowledge, so called informed search
  + Add ``domain-specific information`` to select the best path along which to continue searching
  + 启发式搜索会尽可能地去寻找下一个最好的节点来进行拓展，但无法保证这是最好的节点
  + 启发式搜索特别适用于需要快速的获取一个接近最好结果的解决方法
+ Heuristic function h(n) estimates the 'goodness' of a node n
  + h(n) = estimated cost (or distance) of minimal cost path from n to a goal state
  + All ``domain knowledge`` used in the search is encoded in h(n), which is computable from the current state description.
  + 对于所有的节点存在 h(n) >= 0, 在goal node上, h(n) = 0

### Greedy Search
+ use as an ``evaluation function f(n) = h(n)``, sorting nodes by increasing values of f
+ Selects node to expand believed to be closet to a goal.
+ Greedy search 可能会陷入一个无限循环过程中
+ Not complete and optimal

#### Compared with UCS
+ Cost(n):
  + UCS : g(n): actual path cost thus far
  + Greedy search: h(n) estimate cost to the goal

### A* Algorithm
+ combine the cost ``so far`` and the ``estimated cost to the goal``
  + 把到目前为止的所有cost和预计接下来到终点的花费结合在一起
+ ``f(n) = g(n) + h(n)``
  + ``g``: cost from the initial state to the current state 从起点到现在的所有花费
  + ``h``: 从当前点到目标点位的所有花费
  + h = 0,就会变成UCS搜索，g = 0就会变成greedy search

#### How to estimate h:
+ Optimal and complete: if the heuristic is ``admissible``
+ ``Admissible``: the heuristic must ``never over estimate`` the cost to reach the goal:
  + h(n): a valid ``lower bound`` on cost to the goal
+ For two A* heuristics h1 and h2, if``h1(n) <= h2(n)``, we say ``h2 dominates h1`` or h2 is more ``informed`` than h1
  + A* using h2 will never expand more nodes than A* using h1
+ 永远使用有``higher values but does not over-estimate``的heuristic function

#### Effective Branching Factor
+ Effective branching factor: average number of branches expanded
+ Quality of a heurisitc: ``average effective branching factor``
+ A good heuristic:
  + The ``closer the estimate`` of the heuristic, the better
  + Lower average effective branching factor
  + ``Admissible``

## Lecture 5 Game Playing

### Game Playing - MINIMAX
+ An opponent tries to prevent your win at every move
  + A search method (``Minimax``)
  + ``maximise`` your position whilst ``minimising`` your opponents's
+ ``Utility`` is an abstract measure for the amount of satisfaction you receive from something
  + 我们需要一个方法来衡量一个position的好坏
  + 通常被称作``utility function``
  + Initially this will be a value that describes our position exactly

### Zero-Sum Games
+ Fully ``observable environments``(perfect information), in which ``two agents`` act ``lternately`` 两个agent交替行动
+ Utility values at the end of the game are always ``equal`` or ``opposite``(0+1, 1+0, 1/2+1/2),比如如果一个玩家赢了游戏，另外的玩家就一定会输
+ This opposition between the agents' utility function makes the situation ``adversarial``对抗的

### Components of Game Search
+ ``Initial state``: board position, indication of whose move it is
+ ``A set of operators``: define the legal moves that a player can make
+ ``A terminal test``: determines when the game is over (terminal states)测试游戏是否进入结束的状态
+ ``A utility(payoff) function``: gives a numeric value for the outcom of a game 给出一个数值来表示游戏的结束的结果

#### Game Playing - MINIMAX
+ minimax
  + 两个玩家分别是"MAX" and "MIN"
  + ``untility function(minimax value)`` of a node: the utility (for MAX) of being in the corresponding state (larger values are better for MAX)
+ MAX: take the best move for MAX:
  + Next State: the one with the ``highest utility``, the maximum of its children in the search tree 在搜索树中查找最大值，让MAX保持最高的utility
+ MIN: take the best move for MIN (the worst for MAX)
  + Next State: the ono with the ``lowest utility``, the minimum of its children in the search tree 对于MIN来说，要让MIN赢，则需要让utility越低越好

### Alpha-Beta Pruning(修剪)
+ **Pruning** allow us to ``ignore`` portions of the search tree that make no difference to the final choice
  + 有可能不需要查看所有的节点也可以获得正确的决策
  + Use the idea of ``pruning`` to eliminate(排除) 搜索树的一部分
+ Bounds are stored in terms of two parameters:
  + ``alpha a``: alpha values are stored with each MAX node
    + the ``highest-value`` we have found so far at any choice point along the path of MAX
  + ``beta b``: beta values are stored with each MIN node
    + the ``lowest-value`` we have found so far at any choice point along the path of MIN
  + 当一个MAX节点的alpha值>=任何一个父节点的beta值时，剪掉该节点的所有子节点
  + 当一个MIN节点的alpha值<=任何一个父节点的alpha值时，剪掉该节点的所有子节点

+ BFS 算法可以是有prune nodes吗？
  + 不可以，原因: 减去一个Node D是通过评估Node D下面的树实现的
    + Because the pruning on node D is made by evaluating the tree underneath D
  + 所以只适用于DFS
+ To maximise pruning: first expand the best children
  + use ``heuristics`` for the best-first ordering
  + ``Heuristic evaluation function`` allow us to approximate the true utility of a state without doing a complete search.

### Game classification
+ 1. ``Fully observable``:(chess, checkers and nim)
  + Both players have full and perfect information about the current state of the game.
+ 2. ``Partially Observalble``:
  + Players do not have access to the full ``state of the game`` 玩家无法获取到整个游戏的情况
  + e.g. card games: 没有办法看到你对手的牌
+ ``Deterministic``:
  + There is no element of chance
  + The outcome of making a sequence of moves is entirely determined by the sequencce itself.
+ ``Stochastic``:
  + There is some element of chance 存在一些概率因素
  + 比如throw dice 需要扔色子的游戏

### Generative Adversarial Network GAN
+ Problem: Generative Model
+ Given training data, generate new samples from same distribution

<br>

+ Generative models of time-series data can be used for ``simulation`` and planning 拥有时间序列数据的生成模型可以用于模拟和计划
+ Training generative models can also enable inference of ``latent representations潜在表示`` that can be useful as general features
  + 训练生成模型可以推断潜在的表示形式，这些潜在的表现形式可以用作general features

#### GANs:
+ Instrad of learning a ``explicitly density`` function Pmodel(x)
+ It takes ``game-theoretic 博弈论`` approach: learn to generate from training distribution through 2-player game

#### Two-Player game:
+ ``Generator network`` 生成器，尝试去让判别器无法区分生成的图像
+ ``Discriminator network`` 判别器 尝试去区分真实和生成的图片

<br>
GAN functions

![GANFunction](GAN_function.PNG)

## Lecture 6 Machin Learning & Data Mining

### Supervised learning
+ 给定labels
+ ``Classification``: y is discrete, learn a decision boundary that separates one class from another
+ ``Regression``: y is continuous

### Unsupervised learning
+ given only samples x of the data, infers a function of f describes the ``hidden structure`` of the unlabeled data: ``more of an exploratory/descriptive data analysis``
+ ``Clustering`` 聚类, y is discrete `learn any intrinsic固有的 structure that is present in the data`
+ ``Dimensional Reduction`` 降维, y is continuous. `Discovera lower-dimensional surface on which the data lives`

### Data Mining:
+ Data mining is the exploration and analysis of large quantities of data in order to discover valid, novel, potentially useful, and ultimately understandable patterns in data.
+ Also known as ``Knowledge Discovery in Databases(KDD)``

#### Data Mining Tasks:
+ ``Predictive``: use some variables to predict unknown or future values of other variables
+ ``Descriptive``: Find human-interpretable patterns that describe the data
  + Classification: predictive
  + Clustering: Descriptive
  + Association rule discovery关联规则发现: Descriptive

### CLassfication Methods:
+ ``Regression``: Not flexible enough 不够灵活
+ ``Decision tree``: divide dicision spaace into piecewise constant regions
  + `Internal node`: decision rule on one or more attributes 分支条件
  + `Leaf node`: a predicted class label
  + 优点
    + Reasonable training time 训练花费的时间可以接受
    + Can handle large number of attributes 可以处理相当多的属性
    + Easy to implement
    + Easy to interpret 很容易解释
  + 缺点
    + Simple decision boundaries 决策边界太简单
    + Problems with lots of missing data
    + Cannot handle complicated relationship 很难处理特别复杂的关系
+ ``Neural networks``: partition by non-linear boundaries
  + 善于处理复杂的数据，如 speech, image and handwriting recognition
  + 可以将数据划分为非线性边界 ``partition by nonlinear boundaries``
  + 优点：
    + 更精确，可以学习更复杂的边界
    + 可以除了large number of features
  + 缺点：
    + 很难实现: trial and error for choosing parameters and network structure
    + 训练速度较慢
    + 有可能会过拟合 over-fit
    + 很难进行解释 hard to interpret
+ ``Support vector machines``

#### Classification: K-Nearest Neigbor (KNN)

### Clustering
+ Partition the data so the instances are grouped in similar items by using ``dianstance/similarity`` measure 相似度衡量标准
+ Measure of ``similarity`` between instances
  + ``Euclidean or Manhattan distance`` 欧几里得或曼哈顿距离
    + 欧几里得：空间向量直线距离
    + 曼哈顿：沿着坐标轴运行的距离
  + ``Hamming distance``
  + Other problem specific measures

#### Clustering Methods
+ Partitioning-based clustering
  + ``K-means clustering``
    + Goal: minimise `sum of square of distance` 最小化距离的平方和
      + 距离：每个点和簇中心的距离 each point and centers of the cluster
      + 簇中每一对点之间的距离 each pair of points in the cluster
    + 算法：
      + 初始化 簇中心 Initialize K cluster centors
        + random, first K, K separated points
      + 一直重复直到稳定 Repeat until stabilization
        + Assign each point to closest cluster center 将每一个点附在最近的簇类
        + Generate new cluster centers
        + Adjust clusters by merging or splitting
  + K-medoids clustering
+ ``Density-based clustering``
  + Separate regions of dense points by sparser regions of relatively low density 使用密度较低的稀疏区域将密集点区域分开
  + A cluster: a connected dense component
  + Density: the number of neighbors of a point
  + Can find clusters of arbitrary shape 可以找到任意形状的簇类
+ ``Associate Rules``: market basket analysis
  + discover interesting relations between variables in large databases
  + ``Association rules`` 关联规则: 有60%的买了X和Y的消费者也买了Z
  + ``Sequential patterns`` 序列模式: 40%先买了X的消费者在三周内也会去买Y

## Lecture 7 Nerual Networks

### The first Neural Networks
+ MacCulloch and Pitts produced the first neural network in 1943
+ Consists of:
  + A set of inputs - ``dendrites``
  + A set of resistances/weights -``synapses 突触``
  + A processing element - ``neuron神经元``
  + A single output - ``axon轴突``
    + 神经元的动作是二元的，所以要么fire 要么not fire
+ Neurons in a McCulloch-Pitts network are connected by ``directed, weighted paths``
  + 如果某条path上面的weight是正数，那么这条path则是``excitatory``, otherwise it is ``inhibitory``
    + excitatory path encourage the neuron to fire
    + while inhibitory prevents the neuron from firing
  + Each neuron has a fixed ``threshold``. If the net input in to the neuron is greater than or equal to the threshold, the neuron fires.
    + threshold 极限，输入的净值到达了极限就fire，否则not fire

### Activation Function:
+ Step function (0 - 1)
+ Sign function (-1 - 1)
+ Sigmoid function(0 - 1线性变化) 1/(1+e<sup>-x</sup>)

`X1 XOR X2 = (X1 AND NOT X2) OR (X2 AND NOT X1)`

### Neural Network Architectures

#### Three different classes of network architectures
+ 1. Single-layer feed-forward
+ 2. Multi-layer feed-forward
  + 这两个都是: organized in ``acyclic layers 无环层``
+ 3. recurrent


### Perceptron 感知器
+ Synonym for Single-Layer, Feed-Forwar Network

### Linear Separability
+ Functions which can be separated in this way are called ``Linearly Separable``
+ Only linearly Separable functions can be represented by a single layer NN(perceptron) 只有线性可分的函数才可以被只用一个perceptron表示

#### Epoch:
+ ``Epoch``: The entire training set feed into the neural network. The AND function: an epoch consists of four sets of input(patterns) feed into the network
+ ``Training Value, T``: The value that we require the network to produce.
+ ``Error, Err``: The amount the ouput by the network O differs from the training value T.
+ X<sub>i</sub>: Inputs to the neuron
+ w<sub>i</sub>: Weight from input X to the output
+ LR: the learning rate: how quickly the network converages.
```
While epoch produces an error:
  check next inputs (pattern) from epoch
  Err = T - O
  if Err != 0 then
    wi = wi + LR * X * Err
  End IF
End While 
```
### The learning process: 学习进程
+ Randomly assign initial weights
+ DO:
  + present the network with input
  + Calculate the error value inthe output
  + Adjust the weightings of the inputs according to the error.
+ until no error value exists for all inputs.

<br>

+ 首先随机选择初始的weights
+ 根据现在的weights计算output并与答案进行比较，如果存在错误答案就根据error和learning rate 进行调整，直到找到正确答案。    


## Lecture 8 Probabilistic Reasoning and Bayes' Theorem

### Basic Concepts:
+ If two events A and B are disjoint P(A or B) = P(A) + P(B) 不相交事件
+ P(A<sup>c</sup>) = 1 - P(A)
+ If two events are independent, then the probability of both events happening is the product of probabilities of each event:P(A and B) = P(A)P(B) 相互独立的事件

### Probability Theory
+ p(X,Y) = p(Y|X)p(X)

### Bayes' Theorem
+ p(Y|X) = p(X|Y)p(Y)/p(X)

### Random Variables:
+ A random variable is the basic element of probability, representing an event with some degree of uncertainty as to the event's outcome.

### Joint Probability Distribution
+ Joint Probability Distribution