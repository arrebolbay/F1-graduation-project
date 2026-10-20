# -*- coding: utf-8 -*-
"""单元测试(标准库 unittest, 无需 pytest)。

运行:
    python -m unittest discover -s tests -v
    python -m unittest tests.test_strategy_core -v

只测**不变量**(invariants)—— 那些"一旦被改坏就会让论文数据无声出错"的关系:
DP 目标值与累计曲线末值必须相等、分段闭区间圈数必须累加成 78、清洗链路不删行、
Web 与离线共用同一套常数。
"""
