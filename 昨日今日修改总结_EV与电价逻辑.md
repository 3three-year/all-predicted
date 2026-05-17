# 昨日-今日修改总结：EV 与电价逻辑

## 1. 文档目的

本文档用于总结昨天到今天围绕本项目所做的关键修改，重点包括：

1. EV 放电逻辑
2. EV 充电场景判断逻辑
3. EV 奖励层逻辑
4. EV fallback 保底机制
5. DDQN / PPO / Ablation 的测试链路修复
6. 电价函数简化
7. 成本指标解释

这份文档的目标不是逐行罗列代码差异，而是从“旧逻辑是什么、为什么要改、改成了什么、意义是什么”四个角度做集中说明，方便后续：

1. 查阅
2. 写论文
3. 做答辩
4. 继续迭代代码

---

## 2. EV 放电逻辑：从固定绝对阈值改成净负荷相对判断

### 2.1 旧逻辑

项目原先 EV 放电触发大致采用以下固定阈值逻辑：

```text
pv_output < 1.0
current_load > 0.8
ev_load.soc > 0.4
```

然后再计算：

```text
peak_reduction = min(discharge_limit, (current_load - 0.8) * 0.5)
target_power = target_power - peak_reduction
```

### 2.2 旧逻辑存在的问题

主要问题有 4 点：

1. `pv_output < 1.0` 是固定绝对阈值
2. `current_load > 0.8` 也是固定绝对阈值
3. 不同日型、不同光伏倍率下不稳定
4. 放电逻辑没有直接反映“系统净负荷压力”

因此，旧逻辑更像“工程经验触发”，不够稳，也不够适合论文表达。

### 2.3 新逻辑

现在已改为基于净负荷超额程度的连续削峰逻辑。

首先定义净负荷：

```text
L_net = current_load - pv_output
```

再定义当天净负荷参考量：

```text
L_base = day_mean_net_load
L_peak = day_peak_net_load
```

其中：

```text
day_mean_net_load = mean(day_load_profile - day_pv_profile)
day_peak_net_load = max(day_load_profile - day_pv_profile)
```

### 2.4 放电触发条件

采用如下温和门槛：

```text
L_net > (1 + eta_L) * L_base
且 SOC > 0.50
```

当前参数取值为：

```text
eta_L = 0.10
SOC_dis_threshold = 0.50
```

### 2.5 放电补偿量

放电修正项写成：

```text
Delta_P_peak = - min(discharge_limit, k_discharge * |P_ev_min| * net_load_ratio)
```

其中：

```text
net_load_ratio = clip((L_net - L_base) / (L_peak - L_base), 0, 1)
```

当前参数：

```text
k_discharge = 0.50
```

### 2.6 修改后的意义

修改后的放电逻辑不再是：

```text
低光伏 + 高负荷 -> 直接放电
```

而是：

```text
净负荷越高于当天基准净负荷，放电越强
```

这意味着：

1. 更符合削峰本意
2. 不同日型下更稳
3. 更适合论文表述

---

## 3. EV 充电场景判断：从固定 `pv_output` 阈值改成 `pv_ratio`

### 3.1 旧逻辑

原先 EV 充电场景划分为：

```text
高光伏：pv_output >= 3.0
中光伏：1.0 <= pv_output < 3.0
低光伏：pv_output < 1.0
```

### 3.2 问题

这类固定绝对阈值在以下场景下不稳：

1. 晴天与阴天
2. 夏季与冬季
3. 1.0 倍、1.5 倍、2.0 倍光伏数据版本

也就是说，同样的 `3.0`，在不同日期中不代表同样的“高光伏”。

### 3.3 新逻辑

改成按照当天峰值比例判断：

```text
day_pv_peak = max(day_pv_profile)
pv_ratio = pv_output / max(day_pv_peak, 1e-6)
```

再划分为：

```text
高光伏：pv_ratio >= 0.50
中光伏：0.15 <= pv_ratio < 0.50
低光伏：pv_ratio < 0.15
```

### 3.4 修改后的意义

这一步的核心思想是：

```text
不是看当前光伏绝对值有多大，而是看它占当天这一天光伏能力的比例有多高
```

其优点是：

1. 和当天真实情况绑定
2. 可适配不同光伏倍率数据
3. 论文里更容易解释

---

## 4. EV 奖励层：与 `pv_ratio` 判断统一

### 4.1 旧奖励逻辑

奖励层里原先仍然有固定阈值判断，例如：

1. `pv_output > 5.0` 时给高光伏充电奖励
2. `pv_output < 1.0` 时给低光伏充电惩罚
3. `pv_output < 1.5` 时给傍晚低光伏惩罚

### 4.2 问题

这会造成：

1. 决策层按 `pv_ratio` 判断
2. 奖励层却仍按固定绝对值判断

从而导致策略训练时前后标准不一致。

### 4.3 新奖励逻辑

现在奖励层改成使用 `reward_pv_ratio`：

1. 高光伏奖励：`reward_pv_ratio >= 0.50`
2. 低光伏惩罚：`reward_pv_ratio < 0.15`
3. 傍晚低光伏惩罚：`reward_pv_ratio < 0.10`

### 4.4 修改后的意义

这一步的意义是：

1. 决策层和奖励层口径统一
2. 不再出现“动作逻辑是一套，奖励逻辑又是另一套”的问题

---

## 5. EV 放电 SOC 阈值统一

### 5.1 旧情况

之前代码和文档中曾同时出现：

1. `SOC >= 0.5`
2. `SOC > 0.4`

这导致口径不统一。

### 5.2 新情况

目前统一为：

```text
SOC > 0.50
```

用于 EV 放电削峰门槛。

### 5.3 修改后的意义

统一后更便于：

1. 文档描述
2. 答辩解释
3. 代码维护

---

## 6. EV 目标功率：从分支工程规则整理成半结构化公式

### 6.1 旧逻辑

旧逻辑主要通过 if-else 分支直接决定 `target_power`：

1. 高光伏时一套逻辑
2. 中光伏时再结合 `slack`
3. 低光伏时决定等待还是兜底补电

### 6.2 问题

这种写法：

1. 工程上能跑
2. 但理论表达不规整
3. 不利于论文写法

### 6.3 新逻辑

现在将 EV 目标功率显式写成：

```text
target_power_raw = P_task + Delta_P_pv + Delta_P_prog + Delta_P_peak
```

其中：

#### `P_task`

任务底线项，对应：

```text
required_power
```

表示：

```text
为了在剩余时间内完成剩余任务，当前最少应该充多少
```

#### `Delta_P_pv`

光伏修正项，表示：

```text
当前光伏越好，越鼓励额外多充一点
```

通常为 `0` 或正值。

#### `Delta_P_prog`

进度补偿项，表示：

```text
当前进度越落后，越应额外补一点充电功率
```

通常为 `0` 或正值。

#### `Delta_P_peak`

削峰修正项，表示：

```text
当前净负荷越高，越把 EV 目标功率往放电方向拉
```

通常为 `0` 或负值。

### 6.4 注意

最终执行功率不是“四项裸相加后直接执行”，而是：

```text
target_power_raw
-> 任务门控 / 可用性判断
-> SOC 与功率约束
-> 平滑处理
-> target_power
```

### 6.5 修改后的意义

这一步的本质是：

```text
把原来散在 if-else 中的 EV 目标功率形成机制，显式整理成了结构化表达
```

这样更利于：

1. 论文公式化表达
2. 后续参数调试
3. 逻辑解释

---

## 7. fallback：改成显式开关，并默认关闭

### 7.1 fallback 的定义

fallback 是：

```text
当主策略没有选择 EV 动作 action == 3 时，环境对 DDQN / PPO 的保底补电机制
```

它不是主策略本身，而是环境层的工程保底逻辑。

### 7.2 修改内容

在 `CountrysideEnv` 中新增：

```python
enable_ev_fallback=False
```

并保存为：

```python
self.enable_ev_fallback = enable_ev_fallback
```

fallback 分支改为：

```python
if self.enable_ev_fallback and action != 3 and self.algo in ["dueling_dqn", "ppo"]:
```

### 7.3 意义

现在可以明确区分：

#### 研究对比模式

```text
enable_ev_fallback = False
```

默认关闭，适合更公平的算法比较。

#### 工程保底模式

```text
enable_ev_fallback = True
```

适合演示或保底运行。

---

## 8. DDQN / PPO 测试阶段显式打开 fallback

### 8.1 背景

此前出现过：

1. DDQN 的 EV 曲线全 0
2. PPO 的 EV 曲线全 0

原因是它们自己很少主动选 `action == 3`。

### 8.2 修改内容

在：

1. `ddqn_trainer.py`
2. `ppo_trainer.py`

的测试环境创建中，显式传入：

```python
enable_ev_fallback=True
```

### 8.3 意义

这样：

1. DDQN / PPO 测试时有了 EV 保底补电
2. EV 图中不再全 0

但需要注意：

```text
这表示“带工程保底的测试结果”，不等于纯策略自身学会了调 EV
```

---

## 9. `ev_action_encouragement`：从白给奖励改成只对 `action == 3` 生效

### 9.1 旧逻辑问题

旧逻辑中，只要当天还有 EV 任务，就可能得到这部分奖励，而不管是否真的选择了 `action == 3`。

这会导致：

```text
不选 EV 动作也能拿到 EV 奖励
```

从而削弱策略学习 EV 动作的动机。

### 9.2 新逻辑

现在改为：

```text
只有在：
- reference_ev_energy > 0
- remaining_ev_energy > 0
- 且 action == 3
时，才给 ev_action_encouragement
```

### 9.3 意义

这一步是为了让 DDQN / PPO 真正学会：

```text
如果想拿 EV 奖励，就必须主动选择 action == 3
```

---

## 10. Ablation：补上 EV 负荷采集与绘图

### 10.1 旧问题

Ablation 原先测试阶段没有记录：

```text
env_adqn.ev_load.current_power
```

所以：

1. 没有 `ablation_ev_loads.npy`
2. EV 图中也显示不出来 Ablation

### 10.2 修改内容

在 `ddqn_trainer.py` 的 Ablation `test()` 中补了：

1. `daily_ev_loads`
2. `all_ev_loads`
3. 每步记录 `env_adqn.ev_load.current_power`
4. 保存 `ablation_ev_loads.npy`
5. 返回给 `train_simplified.py`

### 10.3 意义

现在 Ablation 已经可以正常显示在：

1. `06a_典型日1EV负荷调度对比`
2. `06b_典型日2EV负荷调度对比`

---

## 11. 电价函数：从动态反馈型电价改成固定分时电价

### 11.1 旧设计

原先电价函数大致是：

```text
price = base_price
      * seasonal_factor
      * day_factor
      * load_factor
      * pv_factor
      * soc_factor
```

其中：

1. `day_factor`
2. `load_factor`
3. `pv_factor`
4. `soc_factor`

都会使“算法调度结果”反过来影响算法自己面对的电价环境。

### 11.2 问题

这会造成：

```text
算法不仅改变负荷，还改变自己所面对的价格
```

从而使成本比较不够公平。

### 11.3 新设计

现已改成只保留固定分时电价：

1. 峰时：`8-11`, `18-22`
2. 谷时：`23-7`
3. 平时：其余时段取峰谷均值

即：

```text
price = base_price
```

### 11.4 意义

现在所有算法面对的是：

```text
同一张固定分时电价表
```

这样更适合：

1. 成本公平比较
2. 论文结果解释

---

## 12. 成本指标本身没有改，但解释已经更清楚

### 12.1 当前成本计算方式

在 `train_simplified.py` 中，典型日成本仍然按：

```text
cost = Σ [ max(0, load(t) - pv(t)) × price(t) × 0.25 ]
```

计算。

### 12.2 含义

这表示：

1. 每个时刻算法调度柔性负荷和储能
2. 光伏抵消部分负荷
3. 储能实现能量时移
4. 得到净购电负荷
5. 乘电价和时长
6. 累加为总购电成本

### 12.3 理论结论

这条理论链是正确的。

也就是说，当前成本高低的差异，不是成本公式本身的问题，而是：

1. 各算法调出来的负荷轨迹不同
2. 光伏利用方式不同
3. 储能调度轨迹不同

---

## 13. 当前已经恢复的 EV 关键逻辑

截至目前，和 EV 直接相关的关键逻辑已经恢复为：

1. `enable_ev_fallback` 开关：已恢复
2. EV 放电：净负荷判断版，已恢复
3. EV 充电：`pv_ratio` 判断版，已恢复
4. EV 奖励层：相对光伏判断，已恢复
5. `ev_action_encouragement`：只对 `action == 3` 生效，已恢复
6. EV 目标功率：四项结构化版本，已恢复

---

## 14. 仍需注意的事项

虽然 EV 关键逻辑已经恢复，但项目中仍有一些**其他模块**保留着旧的绝对阈值风格，例如：

1. 空调逻辑中的部分 `pv_output < 1.0`
2. 可平移负荷逻辑中的旧绝对阈值
3. 某些总柔性负荷相关的旧写法

这些目前还没有统一按 EV 这套思路重构。

---

## 15. 最终总结

昨天到今天的整体修改方向可以概括为：

```text
把 EV 相关逻辑从“固定绝对阈值 + 工程分支规则主导”的写法，
逐步改成“基于净负荷相对超额、基于相对光伏比例、具备结构化目标功率表达、
并且奖励与测试机制更统一”的版本；
同时把电价环境简化成固定分时电价，以便让算法成本比较更公平、更容易解释。
```

这套修改的核心目标是：

1. 提高逻辑稳定性
2. 提高代码可解释性
3. 提高论文表达质量
4. 提高算法成本比较的公平性

