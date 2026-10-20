#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
策略层不变量测试(strategy_core)
================================

这些断言对应 2026-10 那次"同一份数据两个最优策略"的复盘结论 —— 每一个都曾经
被改坏过,而且改坏时**没有任何报错**,只是悄悄换了结论:

1. :meth:`TestLapTime.test_peak_normalised` —— 圈速必须按各自 peak 归一,
   否则"拟合水平差异"会被重复算进配方偏移(曾导致 C3 被算成最快配方);
2. :meth:`TestStints.test_closed_interval` —— 分段是闭区间,``laps = end - start + 1``
   (旧图用 ``range(start, end)`` 逐圈累加,每段少算 1 圈);
3. :meth:`TestDp.test_objective_matches_curve` —— DP 目标值必须等于累计用时曲线
   的末值,这是上面两条放在一起的对账;
4. :meth:`TestCrossLayer.test_web_constants` —— web/engine.py 与离线共用同一套
   常数(曾各自写死 50 圈上限与硬编码参数)。
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("脚本", "web"):
    _p = os.path.join(ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import strategy_core as core  # noqa: E402


class TestParams(unittest.TestCase):
    """:func:`strategy_core.sanitize_params` 的物理合理性截断。"""

    @classmethod
    def setUpClass(cls):
        cls.params = core.load_pooled_params()

    def test_all_compounds_present(self):
        for comp in core.COMPOUNDS:
            self.assertIn(comp, self.params)

    def test_sanitize_invariants(self):
        for comp, p in self.params.items():
            with self.subTest(comp=comp):
                self.assertGreaterEqual(p["deg"], 0.0, "衰减率不允许为负")
                self.assertGreaterEqual(p["peak"], p["P0"], "峰值不应低于起始值")
                self.assertGreaterEqual(p["t_peak"], 0.0)

    def test_sanitize_repairs_bad_input(self):
        bad = {"P0": 98.0, "peak": 97.0, "t_peak": -3.0, "deg": -0.5}
        fixed = core.sanitize_params(dict(bad))
        self.assertGreaterEqual(fixed["deg"], 0.0)
        self.assertGreaterEqual(fixed["peak"], fixed["P0"])
        self.assertGreaterEqual(fixed["t_peak"], 0.0)



class TestLapTime(unittest.TestCase):
    """性能曲线与圈速公式。"""

    @classmethod
    def setUpClass(cls):
        cls.params = core.load_pooled_params()

    def test_peak_normalised(self):
        """圈速公式必须按各自 peak 归一: (base+offset)/lap_time == perf/peak。"""
        for comp, p in self.params.items():
            peak = float(p["peak"])
            off = 75.0 + core.COMPOUND_OFFSET[comp]
            for age in range(0, 31):
                with self.subTest(comp=comp, age=age):
                    lt = core.lap_time(comp, age, 75.0, p, core.COMPOUND_OFFSET)
                    self.assertAlmostEqual(off / lt, core.perf_pct(age, p) / peak,
                                           places=12)

    def test_degrades_after_peak(self):
        """峰值之后胎龄越大圈速越慢(严格单调),暖胎段之后不得出现回升。"""
        for comp, p in self.params.items():
            if p["deg"] <= 0:
                continue
            start = int(p["t_peak"]) + 1
            seq = [core.lap_time(comp, start + k, 75.0, p, core.COMPOUND_OFFSET)
                   for k in range(40)]
            for a, b in zip(seq, seq[1:]):
                self.assertLess(a, b, f"{comp} 在峰值后应单调变慢")

    def test_fresh_ordering(self):
        """进入衰减段后 C5 最快、C3 最慢(配方偏移方向不能反)。"""
        p = self.params
        t5 = core.lap_time("C5", 1, 75.0, p["C5"], core.COMPOUND_OFFSET)
        t4 = core.lap_time("C4", 1, 75.0, p["C4"], core.COMPOUND_OFFSET)
        t3 = core.lap_time("C3", 17, 75.0, p["C3"], core.COMPOUND_OFFSET)  # C3 峰值在 16 圈
        self.assertLess(t5, t4)
        self.assertLess(t4, t3)


class TestStints(unittest.TestCase):
    """分段提取的闭区间语义。"""

    @classmethod
    def setUpClass(cls):
        cls.params = core.load_pooled_params()
        cls.total, _idx, cls.path = core.solve_dp_best_start(
            cls.params, min_stops=1, max_stops=1)
        cls.stints = core.extract_stints(cls.path)

    def test_one_stop(self):
        self.assertEqual(len(self.stints), 2, "max_stops=1 应恰好两段")

    def test_closed_interval(self):
        for s in self.stints:
            with self.subTest(comp=s["compound"]):
                self.assertEqual(s["laps"], s["end_lap"] - s["start_lap"] + 1)

    def test_laps_sum(self):
        self.assertEqual(sum(s["laps"] for s in self.stints), core.TOTAL_LAPS)
        self.assertEqual(self.stints[0]["start_lap"], 0)
        self.assertEqual(self.stints[-1]["end_lap"], core.TOTAL_LAPS - 1)

    def test_min_stint_laps(self):
        for s in self.stints:
            self.assertGreaterEqual(s["laps"], core.MIN_STINT_LAPS)


class TestDp(unittest.TestCase):
    """DP 与累计曲线必须同源。"""

    @classmethod
    def setUpClass(cls):
        cls.params = core.load_pooled_params()

    def test_objective_matches_curve(self):
        for max_stops in (1, 2, None):
            with self.subTest(max_stops=max_stops):
                total, _idx, path = core.solve_dp_best_start(
                    self.params, min_stops=1, max_stops=max_stops)
                xs, ys = core.cumulative_curve(core.extract_stints(path),
                                               params_by_compound=self.params)
                self.assertEqual(xs[-1], core.TOTAL_LAPS)
                self.assertAlmostEqual(ys[-1], total, places=6)

    def test_max_stops_honoured(self):
        for max_stops in (1, 2):
            with self.subTest(max_stops=max_stops):
                _t, _i, path = core.solve_dp_best_start(
                    self.params, min_stops=1, max_stops=max_stops)
                self.assertLessEqual(len(core.extract_stints(path)) - 1, max_stops)

    def test_min_stops_honoured(self):
        total, _i, path = core.solve_dp_best_start(
            self.params, min_stops=2, max_stops=2)
        self.assertTrue(total < float("inf"), "两停方案应当存在")
        self.assertEqual(len(core.extract_stints(path)) - 1, 2)

    def test_two_stops_not_better_than_one(self):
        """放松约束只可能更好或持平(DP 单调性)。"""
        t1, _a, _p1 = core.solve_dp_best_start(self.params, min_stops=1, max_stops=1)
        t2, _b, _p2 = core.solve_dp_best_start(self.params, min_stops=1, max_stops=2)
        self.assertLessEqual(t2, t1 + 1e-9)


class TestCrossLayer(unittest.TestCase):
    """Web 与离线必须共用同一套常数(否则又是一份数据两个结论)。"""

    def test_web_constants(self):
        try:
            import engine
        except Exception as exc:                       # pragma: no cover
            self.skipTest(f"web/engine.py 不可导入: {exc}")
        self.assertEqual(engine.TOTAL_LAPS, core.TOTAL_LAPS)
        self.assertEqual(engine.MIN_STINT_LAPS, core.MIN_STINT_LAPS)
        self.assertEqual(engine.PRERACE_PIT_LOSS, core.PIT_LOSS)
        self.assertEqual(engine.MAX_TYRE_AGE, core.TOTAL_LAPS,
                         "曾写死 50 圈,与离线的'无上限'冲突")

    def test_web_lap_time_matches_core(self):
        """engine 的圈速封装必须与 strategy_core 逐位一致(不得私自改公式)。"""
        try:
            import engine
        except Exception as exc:                       # pragma: no cover
            self.skipTest(f"web/engine.py 不可导入: {exc}")
        params = core.load_pooled_params()
        for comp in core.COMPOUNDS:
            for age in (0, 1, 12, 30):
                with self.subTest(comp=comp, age=age):
                    self.assertEqual(
                        engine.lap_time(comp, age, 75.0, params[comp],
                                        core.COMPOUND_OFFSET),
                        core.lap_time(comp, age, 75.0, params[comp],
                                      core.COMPOUND_OFFSET))
        self.assertEqual(engine.perf_pct(12, params["C4"]),
                         core.perf_pct(12, params["C4"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)


    def test_two_compounds(self):
        """强制"两种配方"规则:分段里至少两种配方。"""
        self.assertGreaterEqual(len({s["compound"] for s in self.stints}), 2)

    def test_format_strategy(self):
        txt = core.format_strategy(self.stints)
        self.assertIn("圈)", txt)
        self.assertEqual(txt.count("→"), len(self.stints) - 1)
