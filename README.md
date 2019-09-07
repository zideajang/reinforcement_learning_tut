# reinforcement_learning_tut

# 核心内容

- 学会如何把生活中具体问题来强化学习来描述
- 

# 什么是强化学习

### 什么是强化学习

一切的学习都是为了解决一个问题，因为问题存在我们才会通过学习找到解决方法。强化学习和我们之前谈到深度学习 CNN 也好，RNN 也好是所有不同的。这里没有训练的数据集，没有通过监督告诉机器什么是好的，什么是不好的。一切都需机器熟悉环境来学习什么是好的，什么是不好的。

有人夸张 AI = RL + DL 我们
什么是强化学习，我们需要理解好一个基础的概念，帮助我们更好理解深度学习，以及这里概念（术语）之间的关系。然后学习根据 state 采取一定行动 action，然后环境根据 action 和当下 state 给学习者 Agent 一个 reword，也就是环境在当前状态下给 Agent 的 action 的一个反馈，这个反馈可能是奖励也可能惩罚。强化学习就要不断更新策略来得到更好 reward，也就是得到更多奖励。

- Agent 学习者，学习观察环境，观察到就是 state 也就是当前的环境一个快照。
- Environment 环境
- State
- Action 是 Agent 根据当前 state 为了获得更高 reward 采取行动
- Reward

### 强化学习应用


* 大家首先会想起天天和自己下棋的 Alpha Go
* 无人驾驶
* 需要通过学习超越人类，

# 经典条件反射

经典条件反射又称巴甫洛夫条件反射，简单就是一摇铃就给 dog 食物，这样当摇铃即使没有 dog 食物，狗也会流口水（这里流口水表示 dog 开心也就是得到 reward）
- state 表示状态例如，也就是一个一个片段，dog 和摇铃就是一个状态，需要花一些心事理解
- value 就是 US 和 CS 建立关系强弱，这个关系强弱是 dog 建立的，所以是主观
- reward 表示 dog 是否开心

| state(s) | value(v） | reward(r) |
| :--- | :----: | ----: |
| 铃，食物 | 馋 v(铃) >0 | 好事 |
| CS US | 期望 | 结果 |

* Conditioned stimuli CS 铃是 dog 并不了解的，也就是不是天生会有好感
* Unconditioned stimuli US 食物就是 US 是 dog 天生就对其有好感的条件刺激

在这里 state 和 reward 是客观的条件，value 是客观的，这里铃并没有 reward，我们需要定义 value 是通过学习可以改变的值。我们不断通过摇铃给 dog 食物，来更新 value 值。value 表示 US 和 CS 之前关系的强弱。value 是 dog 将食物和铃建立关系，所以是主观的东西。



## Rescorla-Wagner model

$V_{cs}$ = $V_{cs}$ + $A \times $($V_{us}$  $ \times $ us - $V_{cs}$  $\times $ cs)
- cs us 为是或否
- A 为学习速率，有关 A 为学习速率应该小于 1 大于 0



我们分析一下模型，$V_{cs}$ 是要摇铃的值，也就是 dog 对摇铃这个状态的反应，$V_{cs}$ 越大说明 dog 听到铃声越高兴。us 和 cs 的值为 0 或 1 ，1 表示出现该状态，如果 us 和 cs 同时为 1 表示摇铃和食物状态同时出现。将参数解释了完之后我们就可以对该公式一目了然。

# 操作性条件反射

我们都看过马戏团表演，给 dog 数字卡片，如果 dog 能够根据主人或观众说出数字找到对的卡片便可以得到奖赏。


| state(s) | action(a) | reward(r) |
| :--- | :----: | ----: |
| 1 -10 数字卡片 | 选卡片 | 食物 |
| CS US | 期望 | 结果 |

Q(s,a)
- 客观（环境）
state action reward
定义 reward episode temporal discount
    * episode 表示一个回合
    * temporal discount 表示时间，表示我们训练对时间敏感度，例如下快棋，如果落子时间过长会得到扣分
- 主观
state value, action value, policy
    * policy 表示如何主观地选择 action

# 评估问题


```python
import pandas as pd
import numpy as np
```

学习是问题相关的，我们学习的最终目的都是为了解决实际的问题，现在问题是摆在我们前面两条路供我们选择，选择过程也就是评估的过程。具体选择那条路是根据走两条路所花费时间长短来决定，其实可能还有其他因素需要考虑。我们在这里将问题简化到只考虑时间，那么时间就是我们的 reward。

评价走新路相比老路的时间是否
- reward 时间
- state 小路和高速，也就是每段路
- action 这里没有 action 选择
- episode 

#### 已知条件
- s1 小路状态 s2 高速路状态
#### 求对两个状态的值
v(s1),v(s2)
#### 公式

$V_{k+1}$ = $V_k$ + a $ \times $ ($R$ - $V_k$)
- a 表示学习
- 
- delta rule

$V_{k+1}$ 表示当前状态的值，在这里也就是本次走完这段路的时间
$V_k$ 上一次的状态的值


通常我们会用取平均值来求解这个问题，我们这里用 numpy 来解决问题。


```python
t1 = np.array([-5,-8,-2,-3])
t2 = np.array([6,9,8,2])
print("s1 小路平均值 {}".format(np.mean(t1)))
print("s2 告诉平均值 {}".format(np.mean(t2)))
```

    s1 小路平均值 -4.5
    s2 告诉平均值 6.25


t1 表示每一次通过小路相对于老路同等距离的多花时间，- 号表示多耗的时间
t2 表示每一次在高速路相对于老快的时间


```python

v1 = 0

temp = []
for r in t1:
    v1 = v1 + 0.5*(r - v1)
    temp.append(v1)
print(temp)
print(v1)
```

    [-2.5, -5.25, -3.625, -3.3125]
    -3.3125



$V_{k+1}$ = $V_k$ + a $\times$ ($R$ - $V_k$)


```python
v2 = 0
temp = []

for r in t2:
    v2 = v2 + 0.5*(r - v2)
    temp.append(v2)
print(temp)
print(v2)
```

    [3.0, 6.0, 7.0, 4.5]
    4.5


平均值和我们的评估问题有什么不同，也就是我们逐步更新方法要比求平均值有什么好处呢。

$V_{k+1}$ = $V_k$ + $\alpha $ * $\Delta$<br/>
$\Delta$ = [R(s) + $\gamma$$\times$V(s') - V(s)]
$\gamma$ 的取值范围，如果 gamma 等于 0 表示我们对未来毫不关心，如果 gamma 等于一个较大的树，就表示我们只在乎明天，小米就是关注未来而华为更看重今天。对于我们普通人我们更关注当前，虽然也注重未来，但是相比于当前，未来显得更加重要。

## 未来依赖

 TD(temporal Difference) learning<br/>
 这是强化学习一个最重要的一个概念，对于我个人来说也是比较难理解的概念。
V($S_t$) = V($S_t$) + $\alpha $ $\times$ [$R_{t+1}$ + $\gamma$(V$S_{t+1}$) - V($S_t$)]
* t 是任意时间


```python
import matplotlib.pyplot as plt
```


```python
def next_s(s):
    if s == 0:
        return 1
    else:
        return -1
def reward(s):
    if s == 0:
        return np.random.normal(-2,1)
    elif s == 1:
        return np.random.normal(5,1)
```

#### next_s 返回下一个 state
这一部分代码表示相对客观，next_s 会返回下一个 state，我们这里问题小路和高速公路的问题，按路段将 state 分为 2 个 state ，state1 表示小路而 state2 表示高速路。当 s 为 0 表示 state1 作为输入那么他的下一个 state 就是 1 表示高速，如果输入是 1 表示高速返回 -1 表示这一轮结束。
#### reward 表示实际耗时，也就是得到奖赏
state1(小路）耗时比较老路比较多所以 reward 是不好的值，这里用负数表示，相反高速路相对较好返回较大值表示。


```python
A = .5
gamma = 1
```

这里学习者比较主观的参数


```python
V = np.zeros(3)
episode = 10
for k in range(episode):
    s = 0
    while s != -1:
        s_new = next_s(s)
        pred_err = reward(s) + gamma*V[s_new] - V[s]
        V[s] = V[s] + A * pred_err
        s = s_new
    print(V)
```

    [-0.56554769  2.67411715  0.        ]
    [0.02175399 4.05324873 0.        ]
    [0.7024955  4.49618146 0.        ]
    [1.35202657 4.447599   0.        ]
    [3.04172881 4.4682914  0.        ]
    [2.56996809 5.20387836 0.        ]
    [3.0908264  4.63129037 0.        ]
    [2.31287349 4.24980114 0.        ]
    [1.90460886 5.26255299 0.        ]
    [3.25739131 4.50842483 0.        ]


实际生活中问题没有那么简单，首先把过程多个态好处，就是扩展性好。

# policy learning

policy 首先 policy 是函数，是函数就需要有输入和输出，输入 state 输出执行 action 的概率，当然可以是一个 action，不过是给个 action 的概率。之前我们是没有选择，选择小路后就必须选择高速路，不过我们现在可以从老路切换到高速也是可能。<br/>
我们选择 action 会决定下一个 state。

#### optimal  

现在我们就进入控制论的领域，


- 目标: 用最少步数走到目的。
- reward reward 是关于 state 函数，reward 是可以有变化的但是不会依赖于学习者，和采用 action。这里我们将目标的 reward 定义为 1 其他都定义为 -1 
- state 每一个格式就是 state，
- action 可以上下左右走
- 已知条件 state action episode
- rward temporal discount
我们首先需要找到客观条件


```python
gsize = [4,4]
gw = np.zeros([gsize[0],gsize[1]],dtype=np.float32)
gw[0,0] = 1
gw[3,3] = 1
print(gw)
```

    [[1. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 1.]]



```python
def state_act(state,action,gsize):
    newstate = state[:]
    if action == 'u':
        newstate[1] = max(0,state[1]-1)
    elif action == 'd':
        newstate[1] = min(gsize[1]-1,state[1] + 1)
    elif action == 'l':
        newstate[0] = max(0,state[0]-1)
    elif action == 'r':
        newstate[0] = min(gsize[0]-1,state[0]+1)
    else:
        raise ValueError("action not valid")
   
    return newstate
```


```python

```


```python
def reward(state, gw):
    if gw[state[0],state[1]] == 1:
        R = 0
    else:
        R = -1
    return R
```


```python
A = .1
gamma = 1
def policy(state):
    actions = ['u','d','l','r']
    return actions[np.random.randint(len(actions))]
```


```python
episode = 100
V = np.zeros_like(gw)
# V[0,0] = 1
# V[3,3] =1 
for k in range(episode):
    s = [2,2]
    while gw[s[0],s[1]] != 1:
        a = policy(s)
        s_new = state_act(s,a,gsize)
        s = s_new
        pred_err = reward(s,gw) + gamma*V[s_new[0],s_new[1]] - V[s[0],s[1]]
        V[s[0],s[1]] = V[s[0],s[1]] + A * pred_err
        s = s_new
    if k %20 == 0:
        print(V)
```

    [[ 0.  -0.1 -0.3 -0.3]
     [ 0.   0.   0.  -0.3]
     [ 0.  -0.1 -0.1 -0.2]
     [ 0.   0.   0.   0. ]]
    [[ 0.        -2.3       -3.6999986 -3.399999 ]
     [-1.7000003 -1.9000003 -2.6999996 -3.199999 ]
     [-1.8000003 -2.6999996 -1.8000003 -3.299999 ]
     [-2.9999993 -4.399998  -3.299999   0.       ]]
    [[ 0.        -5.1999974 -7.799995  -6.9999957]
     [-4.699998  -6.2999964 -6.9999957 -5.8999968]
     [-3.8999984 -6.899996  -5.799997  -5.0999975]
     [-4.399998  -6.899996  -5.699997   0.       ]]
    [[  0.         -6.1999965  -9.6       -10.100002 ]
     [ -5.799997   -8.399996   -8.999998   -7.0999956]
     [ -6.0999966 -10.000002   -7.899995   -5.9999967]
     [ -8.299995  -10.100002   -9.099998    0.       ]]
    [[  0.         -7.9999948 -11.900009  -12.00001  ]
     [ -7.2999954 -11.400007  -12.600012  -11.100006 ]
     [ -9.5       -13.900017  -10.700005   -8.499996 ]
     [-10.500004  -12.500011  -11.100006    0.       ]]


# 
