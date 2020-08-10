# Project report

## Multi Agent DDPG algorithm implementation details

The RL algorithm for the agents used is Deep Deterministic Policy Gradient with shared replay buffer.
This DDPG multi Agents implementation is based on following key principles:-

Policy-Based Methods:-


The Value-Based Methods like Deep Q-Learning are obtaining an optimal policy 𝜋∗ by trying to estimate the optimal action-value function, Policy-Based Methods directly learn the optimal policy.
Besides this simplification another advantage of a Policy-Based Method is the fact that it is able to handle either stochastic or continuous actions.
On the one hand Policy-Based Methods are using the Monte Carlo (MC) approach for the estimate of expected return:

𝐺𝑡=𝑅𝑡+1+𝑅𝑡+2+...+𝑅𝑇, if the discount factor 𝛾=1
As 𝐺𝑡 is estimated with the full trajectory this yields to a high variance, but to a low bias.
On the other hand Value-Based Methods are using the Temporal Difference (TD) approach to estimate the return:

𝐺𝑡=𝑅𝑡+1+𝐺𝑡+1 , if 𝛾=1
Here 𝐺𝑡+1 is the estimated total return an agent will obtain in the next state. As the estimate of 𝐺𝑡 is always depending on the estimate of the next state, the variance of these estimates is low but biased.
The pros of both methods can be combined in one single algorithm namely the Actor-Critic Method.

Actor-Critic Methods
In Actor-Critic Methods one uses two function approximators (usually neural networks) to learn a policy (Actor) and a value function (Critic). The process looks as follows:

1) Observe state 𝑠 from environment and feed into the Actor.
2) The output are action probabilities 𝜋(𝑎|𝑠;𝜃𝜋). Select one action stochastically and feed back to the environment.
3) Observe next state 𝑠′ and reward 𝑟.
4) Use the tuple (𝑠,𝑎,𝑟,𝑠′) for the TD estimate 𝑦=𝑟+𝛾𝑉(𝑠′;𝜃𝑣)
5) Train the Critic by minimizing the loss 𝐿=(𝑦−𝑉(𝑠;𝜃𝑣)2.
6) Calculate the advantage 𝐴(𝑠,𝑎)=𝑟+𝛾𝑉(𝑠′;𝜃𝑣)−𝑉(𝑠;𝜃𝑣).
7) Train the Actor using the advantage.

Deep Deterministic Policy Gradient
The following section refers to [Lillicrap et al., 2016].
Deep Deterministic Policy Gradient (DDPG) combines the Actor-Critic approach with Deep Q-Learning. The actor function  𝜇(𝑠;𝜃𝜇)  gives the current policy. It maps states to continuous deterministic actions. The critic  𝑄(𝑠,𝑎;𝜃𝑞)  on the other hand is used to calculate action values and is learned using the Bellman equation. DDPG is also using a replay buffer and target networks which already helped to improve performance for Deep Q-Learning. In a finite replay buffer tuples of  (𝑠,𝑎,𝑟,𝑠′)  are stored and then batches are sampled from this buffer to apply for network updates. This tackles the issue of correlated tuples arrised from sequentially exploring the environment. Target networks are used to decouple the TD target from the current action value when performing neutwork updates. The target network is a copy of the Actor and Critic Network which are used to calculated the target. One approach is to update the weights of the target networks  𝜃′  with the weights  𝜃  of the Actor and Critic network periodically. An other approach is to perform soft updates:

𝜃′←𝜏𝜃+(1−𝜏)𝜃′  with  𝜏≪1
In order to scale features batch normalization is being applied. This normalizes each dimension across the samples of the minibatch. An other important issue is handling exploration. By adding a noise process  𝑁  an exploration policy  𝜇′  is constructed:

𝜇′(𝑠𝑡)=𝜇(𝑠𝑡;𝜃𝜇,𝑡)+𝑁
The DDPG process looks as follows:
1) Observe state  𝑠  from environment and feed to Actor.
2) Select action  𝑎=𝜇(𝑠;𝜃𝜇)+𝑁  and feed back to environment.
3) Observe next state  𝑠′  and reward  𝑟 .
4) Store transition  (𝑠,𝑎,𝑟,𝑠′)  in replay buffer and sample random minibatch of  𝑛  tuples. Calculate the TD estimate  𝑦=𝑟+𝛾𝑄′(𝑠′,𝜇′(𝑠′;𝜃′𝜇);𝜃′𝑞)
5) Train the Critic by minimizing the loss  𝐿=𝔼[(𝑦−𝑄(𝑠,𝑎;𝜃𝑞))2]
6) Train Actor with policy gradient  𝔼[∇𝜃𝜇𝑄(𝑠,𝑎;𝜃𝑞)|𝑠=𝑠𝑡,𝑎=𝜇(𝑠𝑡;𝜃𝜇)]=𝔼[∇𝑎𝑄(𝑠,𝑎;𝜃𝑞)|𝑠=𝑠𝑡,𝑎=𝜇(𝑠𝑡)∇𝜃𝜇𝜇(𝑠;𝜃𝜇)|𝑠=𝑠𝑡]
7) Update both target networks using soft update

As one see, this is an off-policy algorithm because the policy which is evaluated uses action  𝑎=𝜇′(𝑠′;𝜃′𝜇) . This is different from the policy which selects action  𝑎=𝜇(𝑠;𝜃𝜇)+𝑁 . An other interesting aspect is that the Critic network has only one output node, which is the action value given the state and the action:  𝑄(𝑠,𝑎;𝜃𝑞)  This is different to Deep Q-Learning where the Q-Network is mapping values to every possible (discrete) action node.


#### Actor Model parameters:

**Model architecture**:
- Fully connected layer1 - input: 24 (state size) output: 512 activation : relu
- Fully connected layer2 - input: 512 output: 256 activation : relu
- Fully connected layer3 - input: 256 output: 2 (action size) activation : tanh

- Optimizer : Adam
- Learning rate = 1e-4

#### Critic Model parameters:

**Model arhitecture**:
- Fully connected layer1 - input: 24 (state size) output: 512 activation :  relu/leaky_relu
- Fully connected layer2 - input: **512 (prev layer) + 2 (action size)** output: 256 activation :  relu
- Fully connected layer3 - input: 256 output: 1 output

- Optimizer : Adam
- Learning rate = 3e-4
- weight_decay=0


The above models are defined in **Pytorch**.

#### RL training parameters:

- Uniform weight Distribution Intialization in hidden layers of Actor and Critic Networks.
- Batch Normalization on Hidden Layers.
- Batch size = 1024
- discount rate = 0.99
- Maximum episodes = 30000
- Replay batch size = 512
- Replay  Buffer size = 1e6
- Replay without prioritization
- Update frequency 4
- TAU from  1e-3
- Learning rate 1e-4 for actor and 3e-4 for critic
- Ornstein-Uhlenbeck noise
- 30% dropout for critic Network.



## Results

### Training Plot
RL training showing score vs episodes.
![Score vs Episodes](/train_result_graph.png)

```
Agents scores after 2000 timesteps in Episode 1
[-0.51999999 -0.54999998]
Episode: 1	Max: -0.52	Min: -0.55	Average: -0.53	Cumulative Average: -0.52 in time: 0.02
Agents scores after 2000 timesteps in Episode 2
[-0.67999998 -0.70999998]
Episode: 2	Max: -0.68	Min: -0.71	Average: -0.69	Cumulative Average: -0.60 in time: 40.02
Agents scores after 2000 timesteps in Episode 3
[-0.62999999 -0.75999998]
Episode: 3	Max: -0.63	Min: -0.76	Average: -0.69	Cumulative Average: -0.61 in time: 92.47
Agents scores after 2000 timesteps in Episode 4
[-0.73999998 -0.64999999]
Episode: 4	Max: -0.65	Min: -0.74	Average: -0.69	Cumulative Average: -0.62 in time: 143.48
Agents scores after 2000 timesteps in Episode 5
[-0.63999999 -0.74999998]
Episode: 5	Max: -0.64	Min: -0.75	Average: -0.69	Cumulative Average: -0.62 in time: 210.57
Agents scores after 2000 timesteps in Episode 6
[-0.66999999 -0.71999998]
Episode: 6	Max: -0.67	Min: -0.72	Average: -0.69	Cumulative Average: -0.63 in time: 259.20
Agents scores after 2000 timesteps in Episode 7
[-0.62999999 -0.74999998]
Episode: 7	Max: -0.63	Min: -0.75	Average: -0.69	Cumulative Average: -0.63 in time: 307.21
Agents scores after 2000 timesteps in Episode 8
[-0.70999998 -0.67999998]
Episode: 8	Max: -0.68	Min: -0.71	Average: -0.69	Cumulative Average: -0.64 in time: 355.13
Agents scores after 2000 timesteps in Episode 9
[-0.76999998 -0.44999999]
Episode: 9	Max: -0.45	Min: -0.77	Average: -0.61	Cumulative Average: -0.62 in time: 402.60
Agents scores after 2000 timesteps in Episode 10
[-0.61999999 -0.67999998]
Episode: 10	Max: -0.62	Min: -0.68	Average: -0.65	Cumulative Average: -0.62 in time: 449.92
Agents scores after 2000 timesteps in Episode 11
[-0.71999998 -0.66999999]
Episode: 11	Max: -0.67	Min: -0.72	Average: -0.69	Cumulative Average: -0.62 in time: 498.08
Agents scores after 2000 timesteps in Episode 12
[-0.59999999 -0.77999998]
Episode: 12	Max: -0.60	Min: -0.78	Average: -0.69	Cumulative Average: -0.62 in time: 546.03
Agents scores after 2000 timesteps in Episode 13
[-0.78999998 -0.55999999]
Episode: 13	Max: -0.56	Min: -0.79	Average: -0.67	Cumulative Average: -0.62 in time: 593.90
Agents scores after 2000 timesteps in Episode 14
[-0.51999999  1.12000004]
Episode: 14	Max: 1.12	Min: -0.52	Average: 0.30	Cumulative Average: -0.49 in time: 641.85
Agents scores after 2000 timesteps in Episode 15
[-0.22999998  0.01000002]
Episode: 15	Max: 0.01	Min: -0.23	Average: -0.11	Cumulative Average: -0.46 in time: 689.75
Agents scores after 2000 timesteps in Episode 16
[ 1.81000004  0.67000003]
Episode: 16	Max: 1.81	Min: 0.67	Average: 1.24	Cumulative Average: -0.32 in time: 738.12
Agents scores after 2000 timesteps in Episode 17
[-0.23999999  1.53000004]
Episode: 17	Max: 1.53	Min: -0.24	Average: 0.65	Cumulative Average: -0.21 in time: 786.44
Agents scores after 2000 timesteps in Episode 18
[ 2.86000005  1.77000005]
Episode: 18	Max: 2.86	Min: 1.77	Average: 2.32	Cumulative Average: -0.04 in time: 835.17
Agents scores after 2000 timesteps in Episode 19
[ 2.82000005  1.30000005]
Episode: 19	Max: 2.82	Min: 1.30	Average: 2.06	Cumulative Average: 0.11 in time: 884.43
Agents scores after 2000 timesteps in Episode 20
[ 1.42000003  2.08000005]
Episode: 20	Max: 2.08	Min: 1.42	Average: 1.75	Cumulative Average: 0.21 in time: 933.69
Agents scores after 2000 timesteps in Episode 21
[ 1.95000004  1.73000005]
Episode: 21	Max: 1.95	Min: 1.73	Average: 1.84	Cumulative Average: 0.29 in time: 983.42
Agents scores after 2000 timesteps in Episode 22
[ 2.90000005  2.27000006]
Episode: 22	Max: 2.90	Min: 2.27	Average: 2.59	Cumulative Average: 0.41 in time: 1032.83
Agents scores after 2000 timesteps in Episode 23
[ 2.58000004  2.22000005]
Episode: 23	Max: 2.58	Min: 2.22	Average: 2.40	Cumulative Average: 0.51 in time: 1082.76
Agents scores after 2000 timesteps in Episode 24
[ 3.51000006  0.39000003]
Episode: 24	Max: 3.51	Min: 0.39	Average: 1.95	Cumulative Average: 0.63 in time: 1133.01
Agents scores after 2000 timesteps in Episode 25
[ 3.42000005  0.82000004]
Episode: 25	Max: 3.42	Min: 0.82	Average: 2.12	Cumulative Average: 0.74 in time: 1183.15
Agents scores after 2000 timesteps in Episode 26
[ 3.40000005  1.91000005]
Episode: 26	Max: 3.40	Min: 1.91	Average: 2.66	Cumulative Average: 0.85 in time: 1233.86
Agents scores after 2000 timesteps in Episode 27
[-0.08999998  1.54000004]
Episode: 27	Max: 1.54	Min: -0.09	Average: 0.73	Cumulative Average: 0.87 in time: 1284.21
Agents scores after 2000 timesteps in Episode 28
[ 1.59000004  2.43000005]
Episode: 28	Max: 2.43	Min: 1.59	Average: 2.01	Cumulative Average: 0.93 in time: 1334.65
Agents scores after 2000 timesteps in Episode 29
[ 1.05000003  2.47000005]
Episode: 29	Max: 2.47	Min: 1.05	Average: 1.76	Cumulative Average: 0.98 in time: 1385.05
Agents scores after 2000 timesteps in Episode 30
[ 1.64000004  2.85000006]
Episode: 30	Max: 2.85	Min: 1.64	Average: 2.25	Cumulative Average: 1.04 in time: 1435.68
Agents scores after 2000 timesteps in Episode 31
[ 2.92000005  0.97000004]
Episode: 31	Max: 2.92	Min: 0.97	Average: 1.95	Cumulative Average: 1.10 in time: 1486.34
Agents scores after 2000 timesteps in Episode 32
[ 3.32000005  1.40000004]
Episode: 32	Max: 3.32	Min: 1.40	Average: 2.36	Cumulative Average: 1.17 in time: 1537.00
Agents scores after 2000 timesteps in Episode 33
[ 3.59000006  1.89000005]
Episode: 33	Max: 3.59	Min: 1.89	Average: 2.74	Cumulative Average: 1.25 in time: 1588.20
Agents scores after 2000 timesteps in Episode 34
[ 3.24000005  3.06000006]
Episode: 34	Max: 3.24	Min: 3.06	Average: 3.15	Cumulative Average: 1.30 in time: 1638.77
Agents scores after 2000 timesteps in Episode 35
[ 3.30000006  3.43000006]
Episode: 35	Max: 3.43	Min: 3.30	Average: 3.37	Cumulative Average: 1.37 in time: 1689.59
Agents scores after 2000 timesteps in Episode 36
[ 3.11000005  4.02000007]
Episode: 36	Max: 4.02	Min: 3.11	Average: 3.57	Cumulative Average: 1.44 in time: 1740.48
Agents scores after 2000 timesteps in Episode 37
[ 3.28000006  4.07000007]
Episode: 37	Max: 4.07	Min: 3.28	Average: 3.68	Cumulative Average: 1.51 in time: 1790.99
Agents scores after 2000 timesteps in Episode 38
[ 3.75000006  4.60000007]
Episode: 38	Max: 4.60	Min: 3.75	Average: 4.18	Cumulative Average: 1.59 in time: 1841.71
Agents scores after 2000 timesteps in Episode 39
[ 4.12000006  4.28000007]
Episode: 39	Max: 4.28	Min: 4.12	Average: 4.20	Cumulative Average: 1.66 in time: 1892.56
Agents scores after 2000 timesteps in Episode 40
[ 4.27000007  4.61000007]
Episode: 40	Max: 4.61	Min: 4.27	Average: 4.44	Cumulative Average: 1.73 in time: 1943.53
Agents scores after 2000 timesteps in Episode 41
[ 3.60000006  4.42000007]
Episode: 41	Max: 4.42	Min: 3.60	Average: 4.01	Cumulative Average: 1.80 in time: 1994.19
Agents scores after 2000 timesteps in Episode 42
[ 3.27000006  4.81000008]
Episode: 42	Max: 4.81	Min: 3.27	Average: 4.04	Cumulative Average: 1.87 in time: 2045.18
Agents scores after 2000 timesteps in Episode 43
[ 3.07000005  4.57000008]
Episode: 43	Max: 4.57	Min: 3.07	Average: 3.82	Cumulative Average: 1.93 in time: 2095.98
Agents scores after 2000 timesteps in Episode 44
[ 3.82000006  3.73000007]
Episode: 44	Max: 3.82	Min: 3.73	Average: 3.78	Cumulative Average: 1.98 in time: 2146.56
Agents scores after 2000 timesteps in Episode 45
[ 4.01000006  3.86000007]
Episode: 45	Max: 4.01	Min: 3.86	Average: 3.94	Cumulative Average: 2.02 in time: 2197.40
Agents scores after 2000 timesteps in Episode 46
[ 4.08000007  4.32000007]
Episode: 46	Max: 4.32	Min: 4.08	Average: 4.20	Cumulative Average: 2.07 in time: 2248.03
Agents scores after 2000 timesteps in Episode 47
[ 4.10000006  4.78000008]
Episode: 47	Max: 4.78	Min: 4.10	Average: 4.44	Cumulative Average: 2.13 in time: 2298.62
Agents scores after 2000 timesteps in Episode 48
[ 4.22000007  3.63000007]
Episode: 48	Max: 4.22	Min: 3.63	Average: 3.93	Cumulative Average: 2.17 in time: 2349.58
Agents scores after 2000 timesteps in Episode 49
[ 4.55000007  3.78000007]
Episode: 49	Max: 4.55	Min: 3.78	Average: 4.17	Cumulative Average: 2.22 in time: 2400.48
Agents scores after 2000 timesteps in Episode 50
[ 3.59000006  4.28000007]
Episode: 50	Max: 4.28	Min: 3.59	Average: 3.94	Cumulative Average: 2.26 in time: 2451.18
Agents scores after 2000 timesteps in Episode 51
[ 3.94000007  3.92000007]
Episode: 51	Max: 3.94	Min: 3.92	Average: 3.93	Cumulative Average: 2.30 in time: 2501.73
Agents scores after 2000 timesteps in Episode 52
[ 4.77000007  4.95000008]
Episode: 52	Max: 4.95	Min: 4.77	Average: 4.86	Cumulative Average: 2.35 in time: 2552.07
Agents scores after 2000 timesteps in Episode 53
[ 4.53000007  4.60000007]
Episode: 53	Max: 4.60	Min: 4.53	Average: 4.57	Cumulative Average: 2.39 in time: 2602.86
Agents scores after 2000 timesteps in Episode 54
[ 4.54000007  4.50000007]
Episode: 54	Max: 4.54	Min: 4.50	Average: 4.52	Cumulative Average: 2.43 in time: 2653.78
Agents scores after 2000 timesteps in Episode 55
[ 4.03000006  4.04000007]
Episode: 55	Max: 4.04	Min: 4.03	Average: 4.04	Cumulative Average: 2.46 in time: 2704.23
Agents scores after 2000 timesteps in Episode 56
[ 4.13000007  3.84000007]
Episode: 56	Max: 4.13	Min: 3.84	Average: 3.99	Cumulative Average: 2.49 in time: 2754.51
Agents scores after 2000 timesteps in Episode 57
[ 4.24000007  3.29000006]
Episode: 57	Max: 4.24	Min: 3.29	Average: 3.77	Cumulative Average: 2.52 in time: 2806.14
Agents scores after 2000 timesteps in Episode 58
[ 4.19000007  4.14000007]
Episode: 58	Max: 4.19	Min: 4.14	Average: 4.17	Cumulative Average: 2.55 in time: 2858.23
Agents scores after 2000 timesteps in Episode 59
[ 4.53000007  3.81000006]
Episode: 59	Max: 4.53	Min: 3.81	Average: 4.17	Cumulative Average: 2.58 in time: 2910.41
Agents scores after 2000 timesteps in Episode 60
[ 4.86000007  3.88000007]
Episode: 60	Max: 4.86	Min: 3.88	Average: 4.37	Cumulative Average: 2.62 in time: 2962.24
Agents scores after 2000 timesteps in Episode 61
[ 4.98000007  4.84000007]
Episode: 61	Max: 4.98	Min: 4.84	Average: 4.91	Cumulative Average: 2.66 in time: 3015.02
Agents scores after 2000 timesteps in Episode 62
[ 5.20000008  4.97000008]
Episode: 62	Max: 5.20	Min: 4.97	Average: 5.09	Cumulative Average: 2.70 in time: 3067.02
Agents scores after 2000 timesteps in Episode 63
[ 5.20000008  5.29000008]
Episode: 63	Max: 5.29	Min: 5.20	Average: 5.25	Cumulative Average: 2.74 in time: 3119.11
Agents scores after 2000 timesteps in Episode 64
[ 5.30000008  5.20000008]
Episode: 64	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 2.78 in time: 3170.77
Agents scores after 2000 timesteps in Episode 65
[ 5.18000008  5.20000008]
Episode: 65	Max: 5.20	Min: 5.18	Average: 5.19	Cumulative Average: 2.82 in time: 3221.91
Agents scores after 2000 timesteps in Episode 66
[ 5.20000008  5.30000008]
Episode: 66	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 2.86 in time: 3274.24
Agents scores after 2000 timesteps in Episode 67
[ 5.30000008  5.20000008]
Episode: 67	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 2.89 in time: 3326.54
Agents scores after 2000 timesteps in Episode 68
[ 5.20000008  5.30000008]
Episode: 68	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 2.93 in time: 3379.37
Agents scores after 2000 timesteps in Episode 69
[ 5.30000008  5.20000008]
Episode: 69	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 2.96 in time: 3430.82
Agents scores after 2000 timesteps in Episode 70
[ 5.20000008  5.30000008]
Episode: 70	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 2.99 in time: 3482.71
Agents scores after 2000 timesteps in Episode 71
[ 4.99000007  5.08000008]
Episode: 71	Max: 5.08	Min: 4.99	Average: 5.04	Cumulative Average: 3.02 in time: 3534.75
Agents scores after 2000 timesteps in Episode 72
[ 5.19000008  5.20000008]
Episode: 72	Max: 5.20	Min: 5.19	Average: 5.20	Cumulative Average: 3.05 in time: 3586.56
Agents scores after 2000 timesteps in Episode 73
[ 5.10000008  5.17000008]
Episode: 73	Max: 5.17	Min: 5.10	Average: 5.14	Cumulative Average: 3.08 in time: 3639.56
Agents scores after 2000 timesteps in Episode 74
[ 5.30000008  5.20000008]
Episode: 74	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 3.11 in time: 3691.55
Agents scores after 2000 timesteps in Episode 75
[ 5.10000008  5.20000008]
Episode: 75	Max: 5.20	Min: 5.10	Average: 5.15	Cumulative Average: 3.14 in time: 3743.76
Agents scores after 2000 timesteps in Episode 76
[ 5.30000008  5.20000008]
Episode: 76	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 3.17 in time: 3795.60
Agents scores after 2000 timesteps in Episode 77
[ 5.10000008  5.18000008]
Episode: 77	Max: 5.18	Min: 5.10	Average: 5.14	Cumulative Average: 3.20 in time: 3847.80
Agents scores after 2000 timesteps in Episode 78
[ 5.30000008  5.09000008]
Episode: 78	Max: 5.30	Min: 5.09	Average: 5.20	Cumulative Average: 3.22 in time: 3901.43
Agents scores after 2000 timesteps in Episode 79
[ 5.08000008  5.07000008]
Episode: 79	Max: 5.08	Min: 5.07	Average: 5.08	Cumulative Average: 3.25 in time: 3953.09
Agents scores after 2000 timesteps in Episode 80
[ 5.30000008  5.09000008]
Episode: 80	Max: 5.30	Min: 5.09	Average: 5.20	Cumulative Average: 3.27 in time: 4005.01
Agents scores after 2000 timesteps in Episode 81
[ 5.20000008  4.97000008]
Episode: 81	Max: 5.20	Min: 4.97	Average: 5.09	Cumulative Average: 3.30 in time: 4055.91
Agents scores after 2000 timesteps in Episode 82
[ 5.20000008  5.07000008]
Episode: 82	Max: 5.20	Min: 5.07	Average: 5.14	Cumulative Average: 3.32 in time: 4107.41
Agents scores after 2000 timesteps in Episode 83
[ 5.09000008  5.08000008]
Episode: 83	Max: 5.09	Min: 5.08	Average: 5.09	Cumulative Average: 3.34 in time: 4159.15
Agents scores after 2000 timesteps in Episode 84
[ 5.30000008  5.09000008]
Episode: 84	Max: 5.30	Min: 5.09	Average: 5.20	Cumulative Average: 3.36 in time: 4210.02
Agents scores after 2000 timesteps in Episode 85
[ 5.10000008  5.07000008]
Episode: 85	Max: 5.10	Min: 5.07	Average: 5.09	Cumulative Average: 3.38 in time: 4262.50
Agents scores after 2000 timesteps in Episode 86
[ 5.30000008  5.20000008]
Episode: 86	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 3.41 in time: 4314.30
Agents scores after 2000 timesteps in Episode 87
[ 5.10000008  5.17000008]
Episode: 87	Max: 5.17	Min: 5.10	Average: 5.14	Cumulative Average: 3.43 in time: 4366.78
Agents scores after 2000 timesteps in Episode 88
[ 5.19000008  5.19000008]
Episode: 88	Max: 5.19	Min: 5.19	Average: 5.19	Cumulative Average: 3.45 in time: 4418.69
Agents scores after 2000 timesteps in Episode 89
[ 5.19000008  5.19000008]
Episode: 89	Max: 5.19	Min: 5.19	Average: 5.19	Cumulative Average: 3.47 in time: 4470.27
Agents scores after 2000 timesteps in Episode 90
[ 5.20000008  5.18000008]
Episode: 90	Max: 5.20	Min: 5.18	Average: 5.19	Cumulative Average: 3.49 in time: 4522.64
Agents scores after 2000 timesteps in Episode 91
[ 5.10000008  5.29000008]
Episode: 91	Max: 5.29	Min: 5.10	Average: 5.20	Cumulative Average: 3.51 in time: 4574.37
Agents scores after 2000 timesteps in Episode 92
[ 5.20000008  4.96000008]
Episode: 92	Max: 5.20	Min: 4.96	Average: 5.08	Cumulative Average: 3.52 in time: 4626.06
Agents scores after 2000 timesteps in Episode 93
[ 5.30000008  5.20000008]
Episode: 93	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 3.54 in time: 4676.33
Agents scores after 2000 timesteps in Episode 94
[ 5.09000008  5.09000008]
Episode: 94	Max: 5.09	Min: 5.09	Average: 5.09	Cumulative Average: 3.56 in time: 4727.40
Agents scores after 2000 timesteps in Episode 95
[ 4.85000007  5.19000008]
Episode: 95	Max: 5.19	Min: 4.85	Average: 5.02	Cumulative Average: 3.58 in time: 4779.09
Agents scores after 2000 timesteps in Episode 96
[ 4.76000007  5.06000008]
Episode: 96	Max: 5.06	Min: 4.76	Average: 4.91	Cumulative Average: 3.59 in time: 4830.24
Agents scores after 2000 timesteps in Episode 97
[ 5.30000008  5.08000008]
Episode: 97	Max: 5.30	Min: 5.08	Average: 5.19	Cumulative Average: 3.61 in time: 4881.89
Agents scores after 2000 timesteps in Episode 98
[ 5.19000008  4.96000008]
Episode: 98	Max: 5.19	Min: 4.96	Average: 5.08	Cumulative Average: 3.63 in time: 4932.65
Agents scores after 2000 timesteps in Episode 99
[ 4.88000007  4.50000007]
Episode: 99	Max: 4.88	Min: 4.50	Average: 4.69	Cumulative Average: 3.64 in time: 4983.67
Agents scores after 2000 timesteps in Episode 100
[ 4.88000007  4.41000007]
Episode: 100	Max: 4.88	Min: 4.41	Average: 4.65	Cumulative Average: 3.65 in time: 5034.87
Episodes(Min,Max and Avg scores till now): 100	Max: 5.30	Min: -0.79	Average: 3.65 in time: 5034.87
Agents scores after 2000 timesteps in Episode 101
[ 4.97000008  4.62000007]
Episode: 101	Max: 4.97	Min: 4.62	Average: 4.80	Cumulative Average: 3.66 in time: 5086.30
Agents scores after 2000 timesteps in Episode 102
[ 5.08000008  4.74000007]
Episode: 102	Max: 5.08	Min: 4.74	Average: 4.91	Cumulative Average: 3.68 in time: 5139.00
Agents scores after 2000 timesteps in Episode 103
[ 5.19000008  4.95000008]
Episode: 103	Max: 5.19	Min: 4.95	Average: 5.07	Cumulative Average: 3.69 in time: 5190.62
Agents scores after 2000 timesteps in Episode 104
[ 5.00000007  4.98000007]
Episode: 104	Max: 5.00	Min: 4.98	Average: 4.99	Cumulative Average: 3.70 in time: 5242.61
Agents scores after 2000 timesteps in Episode 105
[ 5.20000008  5.29000008]
Episode: 105	Max: 5.29	Min: 5.20	Average: 5.25	Cumulative Average: 3.72 in time: 5294.42
Agents scores after 2000 timesteps in Episode 106
[ 5.10000008  4.97000008]
Episode: 106	Max: 5.10	Min: 4.97	Average: 5.04	Cumulative Average: 3.73 in time: 5345.58
Agents scores after 2000 timesteps in Episode 107
[ 4.97000008  5.07000008]
Episode: 107	Max: 5.07	Min: 4.97	Average: 5.02	Cumulative Average: 3.75 in time: 5398.19
Agents scores after 2000 timesteps in Episode 108
[ 4.87000007  5.30000008]
Episode: 108	Max: 5.30	Min: 4.87	Average: 5.09	Cumulative Average: 3.76 in time: 5450.51
Agents scores after 2000 timesteps in Episode 109
[ 5.19000008  4.96000008]
Episode: 109	Max: 5.19	Min: 4.96	Average: 5.08	Cumulative Average: 3.77 in time: 5502.96
Agents scores after 2000 timesteps in Episode 110
[ 5.09000008  5.18000008]
Episode: 110	Max: 5.18	Min: 5.09	Average: 5.14	Cumulative Average: 3.79 in time: 5554.27
Agents scores after 2000 timesteps in Episode 111
[ 5.18000008  4.73000007]
Episode: 111	Max: 5.18	Min: 4.73	Average: 4.96	Cumulative Average: 3.80 in time: 5605.80
Agents scores after 2000 timesteps in Episode 112
[ 4.98000007  4.97000008]
Episode: 112	Max: 4.98	Min: 4.97	Average: 4.98	Cumulative Average: 3.81 in time: 5657.35
Agents scores after 2000 timesteps in Episode 113
[ 5.10000008  5.05000008]
Episode: 113	Max: 5.10	Min: 5.05	Average: 5.08	Cumulative Average: 3.82 in time: 5708.72
Agents scores after 2000 timesteps in Episode 114
[ 5.09000008  5.06000008]
Episode: 114	Max: 5.09	Min: 5.06	Average: 5.08	Cumulative Average: 3.83 in time: 5761.65
Agents scores after 2000 timesteps in Episode 115
[ 5.18000008  5.07000008]
Episode: 115	Max: 5.18	Min: 5.07	Average: 5.13	Cumulative Average: 3.84 in time: 5813.24
Agents scores after 2000 timesteps in Episode 116
[ 4.88000007  5.06000008]
Episode: 116	Max: 5.06	Min: 4.88	Average: 4.97	Cumulative Average: 3.85 in time: 5864.92
Agents scores after 2000 timesteps in Episode 117
[ 5.20000008  5.29000008]
Episode: 117	Max: 5.29	Min: 5.20	Average: 5.25	Cumulative Average: 3.87 in time: 5916.42
Agents scores after 2000 timesteps in Episode 118
[ 4.98000007  4.97000008]
Episode: 118	Max: 4.98	Min: 4.97	Average: 4.98	Cumulative Average: 3.88 in time: 5967.78
Agents scores after 2000 timesteps in Episode 119
[ 5.19000008  5.19000008]
Episode: 119	Max: 5.19	Min: 5.19	Average: 5.19	Cumulative Average: 3.89 in time: 6020.23
Agents scores after 2000 timesteps in Episode 120
[ 5.18000008  5.09000008]
Episode: 120	Max: 5.18	Min: 5.09	Average: 5.14	Cumulative Average: 3.90 in time: 6072.35
Agents scores after 2000 timesteps in Episode 121
[ 5.20000008  5.29000008]
Episode: 121	Max: 5.29	Min: 5.20	Average: 5.25	Cumulative Average: 3.91 in time: 6124.70
Agents scores after 2000 timesteps in Episode 122
[ 4.97000008  5.19000008]
Episode: 122	Max: 5.19	Min: 4.97	Average: 5.08	Cumulative Average: 3.92 in time: 6176.03
Agents scores after 2000 timesteps in Episode 123
[ 5.08000008  5.08000008]
Episode: 123	Max: 5.08	Min: 5.08	Average: 5.08	Cumulative Average: 3.93 in time: 6227.63

Environment solved in 118 episodes!	Average Score: 3.93 in time: 6279.60
```

## Discussion

The bigger hidden network structure in Actor and Critic helps in faster convergence and solution of the environment.
I tried hidden layer combination of (512,256), (300,200) and (256,128) and (128,64) and (100,100) hidden node sizes that
showed how the bigger network help in convergence of the reward.
The environment was solved in 123 episodes using (512,256) hidden network combination.
I have tried both the relu and leaky_relu activation function for critic network. The relu network worked best for me and helped
in convergence rather than leaky_relu.


## Ideas for future work

1. Try Prioritised Experience Replay
