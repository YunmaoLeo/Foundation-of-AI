# MENU

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