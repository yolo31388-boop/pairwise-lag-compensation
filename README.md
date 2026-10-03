# 延迟补偿系统

## 项目描述

延迟补偿系统的bug修复项目。核心模块 `network/lag_compensation.py` 存在多个已知bug，需要修复后使测试全部通过。

## 已知问题

- 回滚只回位置不回姿态和头部
- 延迟计算只用ping不考虑抖动
- 命中检测不考虑子弹飞行时间
- 历史记录采样率固定插值不准
- 延迟补偿无合理性校验可伪造

## 项目结构

```
├── network/lag_compensation.py       # 核心模块（带bug）
├── tests/test_lag_compensation.py    # 验收测试（红态）
├── pyproject.toml          # 项目配置
└── README.md               # 说明文档
```

## 运行测试

```bash
python -m pytest tests/test_lag_compensation.py -q
```

修复目标：所有 15 个测试全部通过。

## 技术要求

- 纯Python标准库，不引入第三方依赖
- 不修改测试文件
- 保持现有API接口不变
- 修复所有已知bug
