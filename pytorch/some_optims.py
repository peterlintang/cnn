# SGD 优化器参数说明
# params: 要优化的参数（通常来自 model.parameters()）
# lr: 学习率，控制参数更新的步长，默认 0.01
# momentum: 动量因子，用于加速收敛和减少震荡，默认 0
# weight_decay: L2 正则化系数，用于防止过拟合，默认 0
# dampening: 动量阻尼，控制动量项的计算，默认 0
# nesterov: 是否使用 Nesterov 动量，默认 False
optimizer = optim.SGD(
    params=model.parameters(),
    lr=0.01,           # 学习率
    momentum=0.9,      # 动量因子
    weight_decay=1e-4, # L2 正则化
    nesterov=True      # 启用 Nesterov 动量
)

# Adam 优化器参数说明
# params: 要优化的参数
# lr: 学习率，默认 0.001（推荐值）
# betas: 用于计算梯度和梯度平方的移动平均系数 (beta1, beta2)
#         beta1 控制一阶矩估计（动量），默认 0.9
#         beta2 控制二阶矩估计（方差），默认 0.999
# eps: 数值稳定项，防止除零错误，默认 1e-8
# weight_decay: L2 正则化系数，默认 0
# amsgrad: 是否使用 AMSGrad 变体，默认 False
optimizer = optim.Adam(
    params=model.parameters(),
    lr=0.001,                      # 推荐使用较小的学习率
    betas=(0.9, 0.999),            # 常用的动量参数
    eps=1e-8,                      # 数值稳定项
    weight_decay=1e-4,             # L2 正则化
    amsgrad=False                  # 是否使用 AMSGrad
)

# AdamW 优化器
# 与 Adam 的主要区别：weight_decay 的实现方式不同
# AdamW 的权重衰减更正确，不会影响梯度的计算
optimizer = optim.AdamW(
    params=model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.01,    # 权重衰减系数，通常比 Adam 设置更大
    amsgrad=False
)
# 推荐的配置：AdamW 通常使用 0.01 的 weight_decay
# 而 Adam 通常使用 0.001

# RMSprop 优化器
# 通过除以梯度的指数加权平均来归一化学习率
optimizer = optim.RMSprop(
    params=model.parameters(),
    lr=0.01,               # 学习率
    alpha=0.99,            # 平方梯度的指数衰减率
    eps=1e-8,              # 数值稳定项
    weight_decay=0,        # L2 正则化
    momentum=0,            # 动量因子
    centered=False         # 是否对梯度进行中心化
)

# Adagrad 优化器
# 适合稀疏数据的优化，会对频繁更新的参数使用较小的学习率
optimizer = optim.Adagrad(
    params=model.parameters(),
    lr=0.01,               # 学习率
    lr_decay=0,            # 学习率衰减
    weight_decay=0,       # L2 正则化
    initial_accumulator_value=0  # 初始累积值
)
