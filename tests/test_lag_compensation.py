"""
延迟补偿系统 - 验收测试

运行方式：python -m pytest tests/test_lag_compensation.py -q
共 15 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network.lag_compensation import LagCompensation


class TestLagCompensation:
    """延迟补偿系统测试套件"""

    def setup_method(self):
        """每个测试前的初始化"""
        self.system = LagCompensation()

    # ---- 世界回滚：姿态与头部 ----

    def test_case_01(self):
        """回滚必须恢复玩家当时的姿态，而不是沿用当前姿态"""
        s = self.system
        s.record_state("p1", (0, 0, 0), posture="crouching", timestamp=0.0)
        s.record_state("p1", (0, 0, 0), posture="standing", timestamp=1.0)
        rewound = s.rewind_world(500.0, now=1.0)
        assert rewound["p1"].posture == "crouching"
        assert s._current["p1"].posture == "standing"  # 当前状态不被污染

    def test_case_02(self):
        """回滚后头部位置必须按当时姿态计算：蹲姿头高 1.0 而非站姿 1.7"""
        s = self.system
        s.record_state("p1", (0, 0, 0), posture="crouching", timestamp=0.0)
        s.record_state("p1", (0, 0, 0), posture="crouching", timestamp=1.0)
        state = s.get_rewound_state("p1", 0.5)
        assert state.head_position == pytest.approx((0.0, 1.0, 0.0))
        assert state.head_position[1] != pytest.approx(1.7)

    # ---- 延迟计算：多因素 + 上限 ----

    def test_case_03(self):
        """延迟 = 单向 ping + 抖动 + 服务器处理耗时"""
        s = self.system
        for rtt in (100, 120, 100, 120, 100, 120):
            s.record_ping("p1", rtt)
        for _ in range(10):
            s.update(0.010)  # 每帧 10ms 处理耗时
        # 上报 110 与实测均值 110 一致，不判伪造
        latency = s.calculate_latency("p1", reported_ping_ms=110.0)
        # 110/2 + 抖动(120-100)/2 + 处理10 = 55 + 10 + 10 = 75
        assert latency == pytest.approx(75.0)

    def test_case_04(self):
        """延迟补偿必须有上限：500ms 延迟的玩家不能获得完整补偿"""
        s = self.system
        assert s.calculate_latency("p1", reported_ping_ms=500.0) == 250.0
        assert s.calculate_latency("p1", reported_ping_ms=10000.0) == 250.0
        assert s.calculate_latency("p1", reported_ping_ms=100.0) == 50.0

    def test_case_05(self):
        """伪造 ping 检测：上报值与服务器实测严重不符时改用实测并标记"""
        s = self.system
        for _ in range(5):
            s.record_ping("cheater", 100.0)
        latency = s.calculate_latency("cheater", reported_ping_ms=10.0)
        assert latency == pytest.approx(50.0)  # 用实测 100/2，而非上报的 5
        assert s._flagged.get("cheater") == "ping_spoofing"
        result = s.compensation_check("cheater", "p2", rewind_ms=50.0)
        assert not result["accepted"]
        assert "ping_spoofing" in result["reasons"]

    # ---- 命中注册：飞行时间 / 距离 / 移动目标 ----

    def test_case_06(self):
        """子弹有有限弹速：命中需要飞行时间，100m 目标不能瞬狙"""
        s = self.system
        s.record_state("t1", (0, 0, 100), posture="standing", timestamp=0.0)
        s.record_state("t1", (0, 0, 100), posture="standing", timestamp=2.0)
        result = s.hit_registration("s1", "t1", origin=(0, 0.9, 0),
                                    direction=(0, 0, 1), bullet_speed=100.0,
                                    shot_time=1.0, reported_ping_ms=0.0)
        assert result["hit"]
        assert result["flight_time"] == pytest.approx(1.0, abs=0.05)
        assert result["distance"] == pytest.approx(100.0, abs=5.0)
        assert result["hit_position"][2] == pytest.approx(100.0, abs=5.0)

    def test_case_07(self):
        """超出子弹最大射程的命中必须被拒绝"""
        s = self.system
        s.record_state("t1", (0, 0, 100), posture="standing", timestamp=0.0)
        s.record_state("t1", (0, 0, 100), posture="standing", timestamp=2.0)
        result = s.hit_registration("s1", "t1", origin=(0, 0.9, 0),
                                    direction=(0, 0, 1), bullet_speed=1000.0,
                                    shot_time=1.0, reported_ping_ms=0.0,
                                    max_distance=50.0)
        assert not result["hit"]
        assert result["reason"] == "out_of_range"

    def test_case_08(self):
        """移动目标：考虑目标速度的提前量射击应当命中（扫掠碰撞）"""
        s = self.system
        s.record_state("t1", (0, 0, 100), velocity=(50, 0, 0), timestamp=0.0)
        s.record_state("t1", (100, 0, 100), velocity=(50, 0, 0), timestamp=2.0)
        # 回滚到 t=1.0 时目标在 (50,0,100)，弹速 500 飞行约 0.235s，
        # 期间目标横移约 11.8m，瞄准提前量位置 (61.8, 0.9, 100)
        result = s.hit_registration("s1", "t1", origin=(0, 0.9, 0),
                                    direction=(61.8, 0, 100), bullet_speed=500.0,
                                    shot_time=1.0, reported_ping_ms=0.0)
        assert result["hit"]
        assert result["flight_time"] == pytest.approx(0.235, abs=0.01)

    def test_case_09(self):
        """移动目标：瞄静态位置打高速目标必须打飞（碰撞盒不被拉长）"""
        s = self.system
        s.record_state("t1", (0, 0, 100), velocity=(50, 0, 0), timestamp=0.0)
        s.record_state("t1", (100, 0, 100), velocity=(50, 0, 0), timestamp=2.0)
        # 正对目标回滚位置 (50,0,100) 开火，但弹速有限，
        # 飞行约 0.23s 内目标已横移约 11.6m，必须打飞
        result = s.hit_registration("s1", "t1", origin=(0, 0.9, 0),
                                    direction=(50, 0, 100), bullet_speed=500.0,
                                    shot_time=1.0, reported_ping_ms=0.0)
        assert not result["hit"]
        assert result["reason"] == "missed"

    # ---- 历史记录：自适应采样与插值 ----

    def test_case_10(self):
        """自适应采样率：高速玩家的采样密度显著高于低速玩家"""
        s = self.system
        dt = 1.0 / 120.0
        for i in range(120):
            t = i * dt
            s.record_state("slow", (t, 0, 0), velocity=(1, 0, 0), timestamp=t)
            s.record_state("fast", (30 * t, 0, 0), velocity=(30, 0, 0), timestamp=t)
        slow_samples = len(s._history["slow"])
        fast_samples = len(s._history["fast"])
        assert slow_samples <= 21          # 约 20Hz
        assert fast_samples >= 55          # 约 60Hz
        assert fast_samples > slow_samples * 2

    def test_case_11(self):
        """采样点之间的位置/速度必须线性插值，保证快速移动判定精度"""
        s = self.system
        s.record_state("p1", (0, 0, 0), velocity=(10, 0, 0), timestamp=0.0)
        s.record_state("p1", (10, 0, 0), velocity=(10, 0, 0), timestamp=1.0)
        mid = s.get_rewound_state("p1", 0.5)
        assert mid.position == pytest.approx((5.0, 0.0, 0.0))
        quarter = s.get_rewound_state("p1", 0.25)
        assert quarter.position == pytest.approx((2.5, 0.0, 0.0))

    def test_case_12(self):
        """历史记录按时长裁剪：超出保留窗口的时刻无法回滚"""
        s = LagCompensation({"history_duration": 1.0})
        for i in range(5):
            s.record_state("p1", (float(i), 0, 0), timestamp=float(i))
        assert s.get_rewound_state("p1", 0.5) is None   # 已被裁剪
        assert s.get_rewound_state("p1", 3.5) is not None
        assert s._history["p1"][0].timestamp >= 3.0

    # ---- 补偿校验：合理性与反作弊 ----

    def test_case_13(self):
        """回滚窗口不得超过补偿上限"""
        s = self.system
        result = s.compensation_check("s1", "t1", rewind_ms=500.0)
        assert not result["accepted"]
        assert "rewind_exceeds_cap" in result["reasons"]
        ok = s.compensation_check("s1", "t1", rewind_ms=250.0)
        assert ok["accepted"]

    def test_case_14(self):
        """命中位置必须与目标当前服务器状态 plausible，明显离谱的命中被拒绝"""
        s = self.system
        s.record_state("t1", (0, 0, 0), posture="standing", timestamp=1.0)
        bad = s.compensation_check("s1", "t1", rewind_ms=100.0,
                                   hit_position=(50, 0, 0))
        assert not bad["accepted"]
        assert "implausible_hit" in bad["reasons"]
        good = s.compensation_check("s1", "t1", rewind_ms=100.0,
                                    hit_position=(0, 1.0, 0))
        assert good["accepted"]

    def test_case_15(self):
        """端到端：高延迟射手命中蹲姿移动目标（头部），且通过合理性校验"""
        s = self.system
        # 射手实测 ping 200，上报一致 -> 有效补偿延迟 100ms
        for _ in range(5):
            s.record_ping("s1", 200.0)
        # 目标蹲姿，以 10m/s 沿 -z 移动，t=1.0 时位于 (0,0,40)
        s.record_state("t1", (0, 0, 50), posture="crouching",
                       velocity=(0, 0, -10), timestamp=0.0)
        s.record_state("t1", (0, 0, 40), posture="crouching",
                       velocity=(0, 0, -10), timestamp=1.0)
        # 回滚 100ms -> 目标在 (0,0,41)，蹲姿头高 1.0；弹速 250 迎头射击
        result = s.hit_registration("s1", "t1", origin=(0, 1.0, 0),
                                    direction=(0, 0, 1), bullet_speed=250.0,
                                    shot_time=1.0, reported_ping_ms=200.0)
        assert result["hit"]
        assert result["headshot"]  # 蹲姿头部在 y=1.0，站姿判定会丢失爆头
        assert result["rewind_ms"] == pytest.approx(100.0)
        assert result["flight_time"] == pytest.approx(41.0 / 260.0, abs=1e-3)
        check = s.compensation_check("s1", "t1", rewind_ms=result["rewind_ms"],
                                     hit_position=result["hit_position"],
                                     reported_ping_ms=200.0)
        assert check["accepted"], check["reasons"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
