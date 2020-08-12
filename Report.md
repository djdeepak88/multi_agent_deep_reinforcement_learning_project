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
[-0.53999999 -0.79999998]
Episode: 1	Max: -0.54	Min: -0.80	Average: -0.67	Cumulative Average: -0.54 in time: 0.01
Agents scores after 2000 timesteps in Episode 2
[-0.66999999 -0.71999998]
Episode: 2	Max: -0.67	Min: -0.72	Average: -0.69	Cumulative Average: -0.60 in time: 35.27
Agents scores after 2000 timesteps in Episode 3
[-0.59999999 -0.78999998]
Episode: 3	Max: -0.60	Min: -0.79	Average: -0.69	Cumulative Average: -0.60 in time: 80.76
Agents scores after 2000 timesteps in Episode 4
[-0.25999998 -0.65999999]
Episode: 4	Max: -0.26	Min: -0.66	Average: -0.46	Cumulative Average: -0.52 in time: 126.48
Agents scores after 2000 timesteps in Episode 5
[ 1.78000004  0.12000003]
Episode: 5	Max: 1.78	Min: 0.12	Average: 0.95	Cumulative Average: -0.06 in time: 172.38
Agents scores after 2000 timesteps in Episode 6
[-0.38999998 -0.39999998]
Episode: 6	Max: -0.39	Min: -0.40	Average: -0.39	Cumulative Average: -0.11 in time: 218.63
Agents scores after 2000 timesteps in Episode 7
[-0.68999998 -0.69999998]
Episode: 7	Max: -0.69	Min: -0.70	Average: -0.69	Cumulative Average: -0.20 in time: 264.83
Agents scores after 2000 timesteps in Episode 8
[-0.73999998 -0.64999999]
Episode: 8	Max: -0.65	Min: -0.74	Average: -0.69	Cumulative Average: -0.25 in time: 311.27
Agents scores after 2000 timesteps in Episode 9
[-0.74999998 -0.63999999]
Episode: 9	Max: -0.64	Min: -0.75	Average: -0.69	Cumulative Average: -0.30 in time: 357.15
Agents scores after 2000 timesteps in Episode 10
[-0.65999999 -0.72999998]
Episode: 10	Max: -0.66	Min: -0.73	Average: -0.69	Cumulative Average: -0.33 in time: 403.49
Agents scores after 2000 timesteps in Episode 11
[-0.66999999 -0.71999998]
Episode: 11	Max: -0.67	Min: -0.72	Average: -0.69	Cumulative Average: -0.36 in time: 449.98
Agents scores after 2000 timesteps in Episode 12
[-0.67999998 -0.70999998]
Episode: 12	Max: -0.68	Min: -0.71	Average: -0.69	Cumulative Average: -0.39 in time: 496.73
Agents scores after 2000 timesteps in Episode 13
[-0.79999998 -0.57999999]
Episode: 13	Max: -0.58	Min: -0.80	Average: -0.69	Cumulative Average: -0.40 in time: 544.06
Agents scores after 2000 timesteps in Episode 14
[-0.60999999 -0.76999998]
Episode: 14	Max: -0.61	Min: -0.77	Average: -0.69	Cumulative Average: -0.42 in time: 591.58
Agents scores after 2000 timesteps in Episode 15
[-0.75999998 -0.62999999]
Episode: 15	Max: -0.63	Min: -0.76	Average: -0.69	Cumulative Average: -0.43 in time: 639.05
Agents scores after 2000 timesteps in Episode 16
[-0.62999999 -0.75999998]
Episode: 16	Max: -0.63	Min: -0.76	Average: -0.69	Cumulative Average: -0.44 in time: 686.79
Agents scores after 2000 timesteps in Episode 17
[ 0.09000002 -0.56999998]
Episode: 17	Max: 0.09	Min: -0.57	Average: -0.24	Cumulative Average: -0.41 in time: 734.84
Agents scores after 2000 timesteps in Episode 18
[ 2.50000004 -0.67999998]
Episode: 18	Max: 2.50	Min: -0.68	Average: 0.91	Cumulative Average: -0.25 in time: 782.55
Agents scores after 2000 timesteps in Episode 19
[ 2.61000004 -0.76999998]
Episode: 19	Max: 2.61	Min: -0.77	Average: 0.92	Cumulative Average: -0.10 in time: 830.51
Agents scores after 2000 timesteps in Episode 20
[ 0.68000003 -0.59999998]
Episode: 20	Max: 0.68	Min: -0.60	Average: 0.04	Cumulative Average: -0.06 in time: 878.55
Agents scores after 2000 timesteps in Episode 21
[ 1.79000004  0.20000003]
Episode: 21	Max: 1.79	Min: 0.20	Average: 1.00	Cumulative Average: 0.03 in time: 926.83
Agents scores after 2000 timesteps in Episode 22
[ 0.72000003 -0.72999998]
Episode: 22	Max: 0.72	Min: -0.73	Average: -0.00	Cumulative Average: 0.06 in time: 975.33
Agents scores after 2000 timesteps in Episode 23
[-0.52999998 -0.74999998]
Episode: 23	Max: -0.53	Min: -0.75	Average: -0.64	Cumulative Average: 0.03 in time: 1023.51
Agents scores after 2000 timesteps in Episode 24
[-0.84999998 -0.53999999]
Episode: 24	Max: -0.54	Min: -0.85	Average: -0.69	Cumulative Average: 0.01 in time: 1072.36
Agents scores after 2000 timesteps in Episode 25
[-0.72999998 -0.65999999]
Episode: 25	Max: -0.66	Min: -0.73	Average: -0.69	Cumulative Average: -0.02 in time: 1121.60
Agents scores after 2000 timesteps in Episode 26
[-0.57999999 -0.80999998]
Episode: 26	Max: -0.58	Min: -0.81	Average: -0.69	Cumulative Average: -0.04 in time: 1171.10
Agents scores after 2000 timesteps in Episode 27
[-0.62999999 -0.75999998]
Episode: 27	Max: -0.63	Min: -0.76	Average: -0.69	Cumulative Average: -0.06 in time: 1221.08
Agents scores after 2000 timesteps in Episode 28
[-0.62999999 -0.75999998]
Episode: 28	Max: -0.63	Min: -0.76	Average: -0.69	Cumulative Average: -0.08 in time: 1270.72
Agents scores after 2000 timesteps in Episode 29
[-0.52999999 -0.85999998]
Episode: 29	Max: -0.53	Min: -0.86	Average: -0.69	Cumulative Average: -0.10 in time: 1320.31
Agents scores after 2000 timesteps in Episode 30
[-0.69999998 -0.67999998]
Episode: 30	Max: -0.68	Min: -0.70	Average: -0.69	Cumulative Average: -0.12 in time: 1370.10
Agents scores after 2000 timesteps in Episode 31
[-0.72999998 -0.54999998]
Episode: 31	Max: -0.55	Min: -0.73	Average: -0.64	Cumulative Average: -0.13 in time: 1419.87
Agents scores after 2000 timesteps in Episode 32
[-0.19999998 -0.16999998]
Episode: 32	Max: -0.17	Min: -0.20	Average: -0.18	Cumulative Average: -0.13 in time: 1469.65
Agents scores after 2000 timesteps in Episode 33
[ 0.51000002  2.68000006]
Episode: 33	Max: 2.68	Min: 0.51	Average: 1.60	Cumulative Average: -0.05 in time: 1519.05
Agents scores after 2000 timesteps in Episode 34
[ 0.46000002  1.42000004]
Episode: 34	Max: 1.42	Min: 0.46	Average: 0.94	Cumulative Average: -0.00 in time: 1568.15
Agents scores after 2000 timesteps in Episode 35
[ 2.47000004  2.86000006]
Episode: 35	Max: 2.86	Min: 2.47	Average: 2.67	Cumulative Average: 0.08 in time: 1617.17
Agents scores after 2000 timesteps in Episode 36
[ 3.29000005  3.28000006]
Episode: 36	Max: 3.29	Min: 3.28	Average: 3.29	Cumulative Average: 0.17 in time: 1666.22
Agents scores after 2000 timesteps in Episode 37
[ 2.12000004  4.28000007]
Episode: 37	Max: 4.28	Min: 2.12	Average: 3.20	Cumulative Average: 0.28 in time: 1715.47
Agents scores after 2000 timesteps in Episode 38
[ 3.12000005  3.54000006]
Episode: 38	Max: 3.54	Min: 3.12	Average: 3.33	Cumulative Average: 0.36 in time: 1764.54
Agents scores after 2000 timesteps in Episode 39
[ 3.57000006  2.99000006]
Episode: 39	Max: 3.57	Min: 2.99	Average: 3.28	Cumulative Average: 0.45 in time: 1813.65
Agents scores after 2000 timesteps in Episode 40
[ 3.47000006  3.66000007]
Episode: 40	Max: 3.66	Min: 3.47	Average: 3.57	Cumulative Average: 0.53 in time: 1863.17
Agents scores after 2000 timesteps in Episode 41
[ 3.49000006  3.36000006]
Episode: 41	Max: 3.49	Min: 3.36	Average: 3.43	Cumulative Average: 0.60 in time: 1912.45
Agents scores after 2000 timesteps in Episode 42
[ 5.05000008  2.95000005]
Episode: 42	Max: 5.05	Min: 2.95	Average: 4.00	Cumulative Average: 0.71 in time: 1962.04
Agents scores after 2000 timesteps in Episode 43
[ 4.12000006  4.41000007]
Episode: 43	Max: 4.41	Min: 4.12	Average: 4.27	Cumulative Average: 0.79 in time: 2011.45
Agents scores after 2000 timesteps in Episode 44
[ 3.38000005  3.09000006]
Episode: 44	Max: 3.38	Min: 3.09	Average: 3.24	Cumulative Average: 0.85 in time: 2060.97
Agents scores after 2000 timesteps in Episode 45
[ 2.90000005  3.70000007]
Episode: 45	Max: 3.70	Min: 2.90	Average: 3.30	Cumulative Average: 0.91 in time: 2110.54
Agents scores after 2000 timesteps in Episode 46
[ 4.20000007  5.02000008]
Episode: 46	Max: 5.02	Min: 4.20	Average: 4.61	Cumulative Average: 1.00 in time: 2160.59
Agents scores after 2000 timesteps in Episode 47
[ 5.10000008  5.18000008]
Episode: 47	Max: 5.18	Min: 5.10	Average: 5.14	Cumulative Average: 1.09 in time: 2210.14
Agents scores after 2000 timesteps in Episode 48
[ 5.17000008  5.10000008]
Episode: 48	Max: 5.17	Min: 5.10	Average: 5.14	Cumulative Average: 1.18 in time: 2259.64
Agents scores after 2000 timesteps in Episode 49
[ 4.65000007  4.84000007]
Episode: 49	Max: 4.84	Min: 4.65	Average: 4.75	Cumulative Average: 1.25 in time: 2309.14
Agents scores after 2000 timesteps in Episode 50
[ 4.87000007  4.85000007]
Episode: 50	Max: 4.87	Min: 4.85	Average: 4.86	Cumulative Average: 1.32 in time: 2358.86
Agents scores after 2000 timesteps in Episode 51
[ 5.09000008  5.08000008]
Episode: 51	Max: 5.09	Min: 5.08	Average: 5.09	Cumulative Average: 1.40 in time: 2408.71
Agents scores after 2000 timesteps in Episode 52
[ 5.30000008  5.30000008]
Episode: 52	Max: 5.30	Min: 5.30	Average: 5.30	Cumulative Average: 1.47 in time: 2458.39
Agents scores after 2000 timesteps in Episode 53
[ 5.20000008  5.30000008]
Episode: 53	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 1.54 in time: 2507.83
Agents scores after 2000 timesteps in Episode 54
[ 5.19000008  5.20000008]
Episode: 54	Max: 5.20	Min: 5.19	Average: 5.20	Cumulative Average: 1.61 in time: 2557.01
Agents scores after 2000 timesteps in Episode 55
[ 5.30000008  5.30000008]
Episode: 55	Max: 5.30	Min: 5.30	Average: 5.30	Cumulative Average: 1.68 in time: 2606.29
Agents scores after 2000 timesteps in Episode 56
[ 5.30000008  5.20000008]
Episode: 56	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 1.74 in time: 2655.68
Agents scores after 2000 timesteps in Episode 57
[ 5.20000008  5.19000008]
Episode: 57	Max: 5.20	Min: 5.19	Average: 5.20	Cumulative Average: 1.80 in time: 2705.43
Agents scores after 2000 timesteps in Episode 58
[ 5.20000008  5.30000008]
Episode: 58	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 1.87 in time: 2755.15
Agents scores after 2000 timesteps in Episode 59
[ 5.30000008  5.20000008]
Episode: 59	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 1.92 in time: 2804.63
Agents scores after 2000 timesteps in Episode 60
[ 5.20000008  5.30000008]
Episode: 60	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 1.98 in time: 2853.73
Agents scores after 2000 timesteps in Episode 61
[ 5.19000008  5.10000008]
Episode: 61	Max: 5.19	Min: 5.10	Average: 5.15	Cumulative Average: 2.03 in time: 2903.16
Agents scores after 2000 timesteps in Episode 62
[ 5.30000008  5.20000008]
Episode: 62	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 2.08 in time: 2952.45
Agents scores after 2000 timesteps in Episode 63
[ 5.20000008  5.30000008]
Episode: 63	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 2.14 in time: 3001.22
Agents scores after 2000 timesteps in Episode 64
[ 5.30000008  5.20000008]
Episode: 64	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 2.19 in time: 3049.86
Agents scores after 2000 timesteps in Episode 65
[ 5.20000008  5.29000008]
Episode: 65	Max: 5.29	Min: 5.20	Average: 5.25	Cumulative Average: 2.23 in time: 3098.65
Agents scores after 2000 timesteps in Episode 66
[ 5.30000008  4.97000008]
Episode: 66	Max: 5.30	Min: 4.97	Average: 5.14	Cumulative Average: 2.28 in time: 3147.47
Agents scores after 2000 timesteps in Episode 67
[ 5.20000008  5.18000008]
Episode: 67	Max: 5.20	Min: 5.18	Average: 5.19	Cumulative Average: 2.32 in time: 3196.19
Agents scores after 2000 timesteps in Episode 68
[ 5.20000008  5.18000008]
Episode: 68	Max: 5.20	Min: 5.18	Average: 5.19	Cumulative Average: 2.37 in time: 3244.81
Agents scores after 2000 timesteps in Episode 69
[ 5.30000008  5.20000008]
Episode: 69	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 2.41 in time: 3293.50
Agents scores after 2000 timesteps in Episode 70
[ 5.20000008  4.94000008]
Episode: 70	Max: 5.20	Min: 4.94	Average: 5.07	Cumulative Average: 2.45 in time: 3342.38
Agents scores after 2000 timesteps in Episode 71
[ 4.85000007  4.96000008]
Episode: 71	Max: 4.96	Min: 4.85	Average: 4.91	Cumulative Average: 2.48 in time: 3392.28
Agents scores after 2000 timesteps in Episode 72
[ 5.07000008  5.09000008]
Episode: 72	Max: 5.09	Min: 5.07	Average: 5.08	Cumulative Average: 2.52 in time: 3441.35
Agents scores after 2000 timesteps in Episode 73
[ 5.09000008  5.08000008]
Episode: 73	Max: 5.09	Min: 5.08	Average: 5.09	Cumulative Average: 2.55 in time: 3490.83
Agents scores after 2000 timesteps in Episode 74
[ 5.08000008  4.99000007]
Episode: 74	Max: 5.08	Min: 4.99	Average: 5.04	Cumulative Average: 2.59 in time: 3540.31
Agents scores after 2000 timesteps in Episode 75
[ 5.20000008  5.19000008]
Episode: 75	Max: 5.20	Min: 5.19	Average: 5.20	Cumulative Average: 2.62 in time: 3590.16
Agents scores after 2000 timesteps in Episode 76
[ 5.20000008  5.19000008]
Episode: 76	Max: 5.20	Min: 5.19	Average: 5.20	Cumulative Average: 2.66 in time: 3639.93
Agents scores after 2000 timesteps in Episode 77
[ 5.09000008  5.19000008]
Episode: 77	Max: 5.19	Min: 5.09	Average: 5.14	Cumulative Average: 2.69 in time: 3689.78
Agents scores after 2000 timesteps in Episode 78
[ 5.30000008  5.08000008]
Episode: 78	Max: 5.30	Min: 5.08	Average: 5.19	Cumulative Average: 2.72 in time: 3739.59
Agents scores after 2000 timesteps in Episode 79
[ 5.20000008  5.19000008]
Episode: 79	Max: 5.20	Min: 5.19	Average: 5.20	Cumulative Average: 2.76 in time: 3789.07
Agents scores after 2000 timesteps in Episode 80
[ 5.30000008  5.19000008]
Episode: 80	Max: 5.30	Min: 5.19	Average: 5.25	Cumulative Average: 2.79 in time: 3838.03
Agents scores after 2000 timesteps in Episode 81
[ 5.09000008  4.97000008]
Episode: 81	Max: 5.09	Min: 4.97	Average: 5.03	Cumulative Average: 2.82 in time: 3886.95
Agents scores after 2000 timesteps in Episode 82
[ 5.20000008  5.29000008]
Episode: 82	Max: 5.29	Min: 5.20	Average: 5.25	Cumulative Average: 2.85 in time: 3935.51
Agents scores after 2000 timesteps in Episode 83
[ 5.19000008  5.20000008]
Episode: 83	Max: 5.20	Min: 5.19	Average: 5.20	Cumulative Average: 2.87 in time: 3983.99
Agents scores after 2000 timesteps in Episode 84
[ 5.30000008  5.20000008]
Episode: 84	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 2.90 in time: 4032.77
Agents scores after 2000 timesteps in Episode 85
[ 5.10000008  5.18000008]
Episode: 85	Max: 5.18	Min: 5.10	Average: 5.14	Cumulative Average: 2.93 in time: 4081.36
Agents scores after 2000 timesteps in Episode 86
[ 5.09000008  5.18000008]
Episode: 86	Max: 5.18	Min: 5.09	Average: 5.14	Cumulative Average: 2.96 in time: 4129.69
Agents scores after 2000 timesteps in Episode 87
[ 5.09000008  5.20000008]
Episode: 87	Max: 5.20	Min: 5.09	Average: 5.15	Cumulative Average: 2.98 in time: 4178.03
Agents scores after 2000 timesteps in Episode 88
[ 5.20000008  5.30000008]
Episode: 88	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 3.01 in time: 4226.46
Agents scores after 2000 timesteps in Episode 89
[ 5.20000008  5.19000008]
Episode: 89	Max: 5.20	Min: 5.19	Average: 5.20	Cumulative Average: 3.03 in time: 4274.82
Agents scores after 2000 timesteps in Episode 90
[ 5.30000008  5.30000008]
Episode: 90	Max: 5.30	Min: 5.30	Average: 5.30	Cumulative Average: 3.06 in time: 4323.71
Agents scores after 2000 timesteps in Episode 91
[ 5.09000008  5.20000008]
Episode: 91	Max: 5.20	Min: 5.09	Average: 5.15	Cumulative Average: 3.08 in time: 4372.09
Agents scores after 2000 timesteps in Episode 92
[ 5.19000008  5.20000008]
Episode: 92	Max: 5.20	Min: 5.19	Average: 5.20	Cumulative Average: 3.10 in time: 4420.26
Agents scores after 2000 timesteps in Episode 93
[ 5.09000008  5.09000008]
Episode: 93	Max: 5.09	Min: 5.09	Average: 5.09	Cumulative Average: 3.13 in time: 4469.21
Agents scores after 2000 timesteps in Episode 94
[ 5.30000008  5.30000008]
Episode: 94	Max: 5.30	Min: 5.30	Average: 5.30	Cumulative Average: 3.15 in time: 4518.23
Agents scores after 2000 timesteps in Episode 95
[ 5.20000008  5.30000008]
Episode: 95	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 3.17 in time: 4567.02
Agents scores after 2000 timesteps in Episode 96
[ 5.30000008  5.20000008]
Episode: 96	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 3.19 in time: 4615.72
Agents scores after 2000 timesteps in Episode 97
[ 5.20000008  5.30000008]
Episode: 97	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 3.22 in time: 4664.94
Agents scores after 2000 timesteps in Episode 98
[ 5.20000008  5.19000008]
Episode: 98	Max: 5.20	Min: 5.19	Average: 5.20	Cumulative Average: 3.24 in time: 4713.80
Agents scores after 2000 timesteps in Episode 99
[ 5.20000008  5.08000008]
Episode: 99	Max: 5.20	Min: 5.08	Average: 5.14	Cumulative Average: 3.26 in time: 4762.83
Agents scores after 2000 timesteps in Episode 100
[ 5.19000008  4.74000007]
Episode: 100	Max: 5.19	Min: 4.74	Average: 4.97	Cumulative Average: 3.27 in time: 4812.04
Episodes(Min,Max and Avg scores till now): 100	Max: 5.30	Min: -0.86	Average: 3.27 in time: 4812.04
Agents scores after 2000 timesteps in Episode 101
[ 4.42000007  4.76000007]
Episode: 101	Max: 4.76	Min: 4.42	Average: 4.59	Cumulative Average: 3.29 in time: 4861.07
Agents scores after 2000 timesteps in Episode 102
[ 4.66000007  4.24000007]
Episode: 102	Max: 4.66	Min: 4.24	Average: 4.45	Cumulative Average: 3.30 in time: 4910.44
Agents scores after 2000 timesteps in Episode 103
[ 4.77000007  4.81000008]
Episode: 103	Max: 4.81	Min: 4.77	Average: 4.79	Cumulative Average: 3.32 in time: 4958.71
Agents scores after 2000 timesteps in Episode 104
[ 4.99000007  4.73000007]
Episode: 104	Max: 4.99	Min: 4.73	Average: 4.86	Cumulative Average: 3.33 in time: 5006.72
Agents scores after 2000 timesteps in Episode 105
[ 4.87000007  2.89000005]
Episode: 105	Max: 4.87	Min: 2.89	Average: 3.88	Cumulative Average: 3.35 in time: 5055.03
Agents scores after 2000 timesteps in Episode 106
[ 4.78000007  2.50000005]
Episode: 106	Max: 4.78	Min: 2.50	Average: 3.64	Cumulative Average: 3.36 in time: 5102.96
Agents scores after 2000 timesteps in Episode 107
[ 4.77000007  2.89000005]
Episode: 107	Max: 4.77	Min: 2.89	Average: 3.83	Cumulative Average: 3.37 in time: 5150.93
Agents scores after 2000 timesteps in Episode 108
[ 4.89000007  3.45000006]
Episode: 108	Max: 4.89	Min: 3.45	Average: 4.17	Cumulative Average: 3.39 in time: 5198.62
Agents scores after 2000 timesteps in Episode 109
[ 4.88000007  4.35000007]
Episode: 109	Max: 4.88	Min: 4.35	Average: 4.62	Cumulative Average: 3.40 in time: 5246.45
Agents scores after 2000 timesteps in Episode 110
[ 4.89000007  4.16000007]
Episode: 110	Max: 4.89	Min: 4.16	Average: 4.53	Cumulative Average: 3.42 in time: 5294.39
Agents scores after 2000 timesteps in Episode 111
[ 4.98000007  3.92000007]
Episode: 111	Max: 4.98	Min: 3.92	Average: 4.45	Cumulative Average: 3.43 in time: 5342.16
Agents scores after 2000 timesteps in Episode 112
[ 4.88000007  4.24000007]
Episode: 112	Max: 4.88	Min: 4.24	Average: 4.56	Cumulative Average: 3.44 in time: 5389.86
Agents scores after 2000 timesteps in Episode 113
[ 4.77000007  5.06000008]
Episode: 113	Max: 5.06	Min: 4.77	Average: 4.92	Cumulative Average: 3.46 in time: 5437.64
Agents scores after 2000 timesteps in Episode 114
[ 4.86000007  4.85000007]
Episode: 114	Max: 4.86	Min: 4.85	Average: 4.86	Cumulative Average: 3.47 in time: 5485.30
Agents scores after 2000 timesteps in Episode 115
[ 4.96000008  4.96000008]
Episode: 115	Max: 4.96	Min: 4.96	Average: 4.96	Cumulative Average: 3.48 in time: 5533.12
Agents scores after 2000 timesteps in Episode 116
[ 4.78000007  4.85000007]
Episode: 116	Max: 4.85	Min: 4.78	Average: 4.82	Cumulative Average: 3.49 in time: 5580.85
Agents scores after 2000 timesteps in Episode 117
[ 5.19000008  4.96000008]
Episode: 117	Max: 5.19	Min: 4.96	Average: 5.08	Cumulative Average: 3.51 in time: 5628.35
Agents scores after 2000 timesteps in Episode 118
[ 5.20000008  5.05000008]
Episode: 118	Max: 5.20	Min: 5.05	Average: 5.13	Cumulative Average: 3.52 in time: 5675.88
Agents scores after 2000 timesteps in Episode 119
[ 5.20000008  4.96000008]
Episode: 119	Max: 5.20	Min: 4.96	Average: 5.08	Cumulative Average: 3.54 in time: 5723.54
Agents scores after 2000 timesteps in Episode 120
[ 5.10000008  4.97000008]
Episode: 120	Max: 5.10	Min: 4.97	Average: 5.04	Cumulative Average: 3.55 in time: 5771.32
Agents scores after 2000 timesteps in Episode 121
[ 4.99000007  5.18000008]
Episode: 121	Max: 5.18	Min: 4.99	Average: 5.09	Cumulative Average: 3.56 in time: 5818.81
Agents scores after 2000 timesteps in Episode 122
[ 5.19000008  4.97000008]
Episode: 122	Max: 5.19	Min: 4.97	Average: 5.08	Cumulative Average: 3.58 in time: 5866.32
Agents scores after 2000 timesteps in Episode 123
[ 5.19000008  5.20000008]
Episode: 123	Max: 5.20	Min: 5.19	Average: 5.20	Cumulative Average: 3.59 in time: 5914.13
Agents scores after 2000 timesteps in Episode 124
[ 5.20000008  5.30000008]
Episode: 124	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 3.60 in time: 5961.63
Agents scores after 2000 timesteps in Episode 125
[ 5.20000008  5.20000008]
Episode: 125	Max: 5.20	Min: 5.20	Average: 5.20	Cumulative Average: 3.62 in time: 6009.76
Agents scores after 2000 timesteps in Episode 126
[ 5.30000008  5.20000008]
Episode: 126	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 3.63 in time: 6057.57
Agents scores after 2000 timesteps in Episode 127
[ 5.20000008  5.30000008]
Episode: 127	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 3.64 in time: 6105.84
Agents scores after 2000 timesteps in Episode 128
[ 5.30000008  5.20000008]
Episode: 128	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 3.66 in time: 6154.54
Agents scores after 2000 timesteps in Episode 129
[ 5.20000008  5.30000008]
Episode: 129	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 3.67 in time: 6203.38
Agents scores after 2000 timesteps in Episode 130
[ 5.30000008  5.20000008]
Episode: 130	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 3.68 in time: 6251.66
Agents scores after 2000 timesteps in Episode 131
[ 5.20000008  5.08000008]
Episode: 131	Max: 5.20	Min: 5.08	Average: 5.14	Cumulative Average: 3.69 in time: 6299.55
Agents scores after 2000 timesteps in Episode 132
[ 5.20000008  5.18000008]
Episode: 132	Max: 5.20	Min: 5.18	Average: 5.19	Cumulative Average: 3.70 in time: 6347.46
Agents scores after 2000 timesteps in Episode 133
[ 5.30000008  5.20000008]
Episode: 133	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 3.72 in time: 6395.20
Agents scores after 2000 timesteps in Episode 134
[ 5.20000008  5.19000008]
Episode: 134	Max: 5.20	Min: 5.19	Average: 5.20	Cumulative Average: 3.73 in time: 6442.97
Agents scores after 2000 timesteps in Episode 135
[ 5.09000008  5.08000008]
Episode: 135	Max: 5.09	Min: 5.08	Average: 5.09	Cumulative Average: 3.74 in time: 6491.04
Agents scores after 2000 timesteps in Episode 136
[ 5.20000008  5.30000008]
Episode: 136	Max: 5.30	Min: 5.20	Average: 5.25	Cumulative Average: 3.75 in time: 6538.95
Agents scores after 2000 timesteps in Episode 137
[ 5.30000008  4.96000008]
Episode: 137	Max: 5.30	Min: 4.96	Average: 5.13	Cumulative Average: 3.76 in time: 6588.12
Agents scores after 2000 timesteps in Episode 138
[ 4.98000007  5.07000008]
Episode: 138	Max: 5.07	Min: 4.98	Average: 5.03	Cumulative Average: 3.77 in time: 6637.34
Agents scores after 2000 timesteps in Episode 139
[ 4.96000008  4.76000007]
Episode: 139	Max: 4.96	Min: 4.76	Average: 4.86	Cumulative Average: 3.78 in time: 6686.26
Agents scores after 2000 timesteps in Episode 140
[ 4.99000007  5.07000008]
Episode: 140	Max: 5.07	Min: 4.99	Average: 5.03	Cumulative Average: 3.79 in time: 6735.31

Environment solved in 140 episodes!	Average Score: 3.79 in time: 6784.01
```

## Discussion

The bigger hidden network structure in Actor and Critic helps in faster convergence and solution of the environment.
I tried hidden layer combination of (512,256), (300,200) and (256,128) and (128,64) and (100,100) hidden node sizes that
showed how the bigger hidden network node size help in early convergence towards the higher average reward scores.
The environment was solved in 140 episodes using (512,256) hidden network combination.
I have tried both the relu and leaky_relu activation function for critic network. The relu network worked best for me and helped
in convergence rather than leaky_relu.

## Ideas for future work

1. Try Prioritised Experience Replay
