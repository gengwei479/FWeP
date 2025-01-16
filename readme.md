# Fixed Wing e-Pilot based on Large Language Model

1.轨迹执行误差累积，phased CoT（每一个phase规划出一个clip，理论依据：阶段独立性，200 steps的推导，理论依据：轨迹误差累积与侧翻条件）

Understanding Chain-of-Thought in LLMs through Information Theory



**Aircraft Flight Dynamics and Control**

2.从轨迹数据的统计角度分析，为什么gpt2的效果更好



对比算法用的是PID控制算法（比例控制算法、积分控制算法、微分控制算法）







轨迹$i$的观测序列为$[s^{(i)}_1,\cdots,s^{(i)}_T]$,动作序列为$[a^{(i)}_1,\cdots,a^{(i)}_T]$,经过离散化以后的序列分别为：$[\tilde s^{(i)}_1,\cdots,\tilde s^{(i)}_T]$,$[\tilde a^{(i)}_1,\cdots,\tilde a^{(i)}_T]$,

对于生成任务而言，

$\text{P}(\tilde a^{(i)}_{1:T}|\tilde s^{(i)}_1,\cdots,\tilde s^{(i)}_T)=\prod \limits_{t=1}^T \text{P}(\tilde a^{(i)}_{t+1}|\tilde a^{(i)}_{1:t};\tilde s^{(i)}_1,\cdots,\tilde s^{(i)}_T)$

假设模型的vocabulary size为$C_{\text{voc\_size}}$，误差的度量为:

$loss(\tilde a^{(i)}_{1:T},\hat{\tilde a}^{(i)}_{1:T})=\frac{1}{TC_{\text{voc\_size}}}\sum \limits_{t=1}^{T}(\tilde a^{(i)}_t-\hat{\tilde a}^{(i)}_t)^2$,