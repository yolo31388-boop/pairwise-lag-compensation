"""
延迟补偿系统 - 验收测试

运行方式：python -m pytest tests/test_lag_compensation.py -q
共 15 个测试用例，覆盖：
- 世界回滚包含姿态与头部位置（01-03）
- 延迟计算综合多因素且有上限（04-06）
- 命中检测考虑子弹飞行时间与移动速度（07-09）
- 历史记录采样率自适应且插值准确（10-12）
- 延迟补偿合理性校验与反作弊（13-15）
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network.lag_compensation import LagCompensation


def _record_line(system, player_id, start, end, step, pos_fn, velocity=(0.0, 0.0, 0.0),
                 posture="standing"):
    """沿时间轴记录一系列状态，pos_fn(t) 返回位置"""
    t = start
    while t <= end + 1e-9:
        system.record_state(player_id, pos_fn(t), posture=posture,
                            velocity=velocity, timestamp=t)
        t += step


class TestLagCompensation:
    """延迟补偿系统测试套件"""

    def setup_method(self):
        """每个测试前的初始化"""
        self.system = LagCompensation()

    # ---------------- 世界回滚：姿态与头部 ----------------

    def test_case_01(self):
        """回滚必须恢复玩家姿态（站/蹲），不能只回位置"""
        self.system.record_state("p1", (0, 0, 0), posture="standing", timestamp=0.0)
        self.system.record_state("p1", (0, 0, 0), posture="crouching", timestamp=1.0)

        rewound = self.system.rewind_world(["p1"], 1.0)
        assert rewound["p1"].posture == "crouching"

        rewound = self.system.rewind_world(["p1"], 0.0)
        assert rewound["p1"].posture == "standing"

    def test_case_02(self):
        """回滚必须恢复头部位置，蹲下时头部低于站立时"""
        self.system.record_state("p1", (5, 0, 5), posture="standing", timestamp=0.0)
        self.system.record_state("p2", (5, 0, 5), posture="crouching", timestamp=0.0)

        rewound = self.system.rewind_world(["p1", "p2"], 0.0)
        standing_head = rewound["p1"].head_position
        crouching_head = rewound["p2"].head_position
        assert standing_head[1] == pytest.approx(1.7)
        assert crouching_head[1] == pytest.approx(1.1)
        assert crouching_head[1] < standing_head[1]

    def test_case_03(self):
        """回滚到两个样本之间时必须线性插值位置"""
        self.system.record_state("p1", (0, 0, 0), velocity=(10, 0, 0), timestamp=0.0)
        self.system.record_state("p1", (10, 0, 0), velocity=(10, 0, 0), timestamp=1.0)

        rewound = self.system.rewind_world(["p1"], 0.5)
        assert rewound["p1"].position[0] == pytest.approx(5.0)

    # ---------------- 延迟计算：多因素 + 上限 ----------------

    def test_case_04(self):
        """延迟必须综合 ping、网络抖动与服务器处理耗时"""
        for ping in (0.10, 0.12, 0.08, 0.10):
            self.system.report_ping("p1", ping)
        report = self.system.calculate_latency("p1", client_ping=0.10,
                                               server_processing_time=0.01)
        assert report["jitter"] == pytest.approx(0.01)
        assert report["latency"] == pytest.approx(0.10 + 0.01 + 0.01)
        assert report["latency"] > 0.10  # 不能只用客户端 ping

    def test_case_05(self):
        """延迟必须有上限，500ms 高延迟玩家不能获得完整补偿"""
        report = self.system.calculate_latency("p1", client_ping=0.5)
        assert report["capped"] is True
        assert report["latency"] == pytest.approx(self.system.config["max_latency"])
        assert report["latency"] < 0.5

    def test_case_06(self):
        """客户端伪造高 ping 时必须按服务器实测值计算并标记"""
        report = self.system.calculate_latency("p1", client_ping=0.4,
                                               server_measured_ping=0.1)
        assert report["forged_ping"] is True
        assert report["effective_ping"] == pytest.approx(0.1)
        assert report["latency"] == pytest.approx(0.1 + 0.005)

    # ---------------- 命中检测：飞行时间与移动速度 ----------------

    def _moving_target(self):
        # 目标位于 x=50，沿 z 轴以 8 m/s 移动，t=1 时位于原点正前方
        _record_line(self.system, "target", 0.0, 3.0, 0.25,
                     lambda t: (50.0, 0.0, 8.0 * (t - 1.0)), velocity=(0, 0, 8))

    def test_case_07(self):
        """命中检测必须考虑子弹飞行时间：瞄当前位置打不中移动目标"""
        self._moving_target()
        origin = (0.0, 0.9, 0.0)
        # 瞄准回滚时刻目标所在位置（未提前量）-> 应脱靶
        result = self.system.hit_registration("shooter", "target", origin,
                                              (50.0, 0.0, 0.0), 100.0, 1.0)
        assert result["hit"] is False
        assert result["flight_time"] > 0.4  # 子弹飞行约 0.5 秒
        # 瞄准飞行时间后的提前量位置 -> 应命中
        result = self.system.hit_registration("shooter", "target", origin,
                                              (50.0, 0.0, 4.0), 100.0, 1.0)
        assert result["hit"] is True

    def test_case_08(self):
        """超出子弹最大射程的射击必须判定未命中"""
        _record_line(self.system, "target", 0.0, 2.0, 0.5,
                     lambda t: (400.0, 0.0, 0.0))
        result = self.system.hit_registration("shooter", "target", (0, 0.9, 0),
                                              (1, 0, 0), 1000.0, 1.0)
        assert result["hit"] is False
        assert "射程" in result["reason"]

    def test_case_09(self):
        """高速目标的碰撞盒不得被拉长：偏离弹道 0.5m 必须脱靶"""
        self._moving_target()
        # 瞄准提前量后方 0.5m（若碰撞盒随速度拉伸则会误判命中）
        result = self.system.hit_registration("shooter", "target", (0, 0.9, 0),
                                              (50.0, 0.0, 3.5), 100.0, 1.0)
        assert result["hit"] is False

    # ---------------- 历史记录：自适应采样与插值 ----------------

    def test_case_10(self):
        """采样率必须自适应：高速移动玩家的样本密度显著高于静止玩家"""
        t = 0.0
        while t <= 1.0 + 1e-9:
            self.system.record_state("fast", (10.0 * t, 0, 0),
                                     velocity=(10, 0, 0), timestamp=t)
            self.system.record_state("slow", (0, 0, 0), timestamp=t)
            t += 0.01
        fast_count = self.system.history_length("fast")
        slow_count = self.system.history_length("slow")
        assert fast_count > 60
        assert slow_count < 40
        assert fast_count > slow_count * 2

    def test_case_11(self):
        """历史记录必须保留超过 1 秒，支持更大的回滚窗口"""
        _record_line(self.system, "p1", 0.0, 2.0, 0.1,
                     lambda t: (1.0 * t, 0, 0), velocity=(1, 0, 0))
        state = self.system.get_state_at("p1", 0.5)
        assert state is not None
        assert state.timestamp == pytest.approx(0.5)
        assert state.position[0] == pytest.approx(0.5)

    def test_case_12(self):
        """快速移动目标的插值必须准确"""
        _record_line(self.system, "p1", 0.0, 1.0, 0.05,
                     lambda t: (10.0 * t, 0, 0), velocity=(10, 0, 0))
        state = self.system.get_state_at("p1", 0.525)
        assert state.position[0] == pytest.approx(5.25, abs=1e-6)

    # ---------------- 补偿校验：合理性与反作弊 ----------------

    def _legit_hit(self):
        """构造一次完全合法的命中及其延迟报告"""
        _record_line(self.system, "target", 0.0, 2.0, 0.1,
                     lambda t: (10.0, 0.0, 0.0))
        report = self.system.calculate_latency("shooter", client_ping=0.08,
                                               server_measured_ping=0.08)
        hit = self.system.hit_registration("shooter", "target", (0, 0.9, 0),
                                           (10.0, 0.0, 0.0), 100.0, 1.0,
                                           shooter_latency=report["latency"])
        assert hit["hit"] is True
        return report, hit

    def test_case_13(self):
        """回滚命中位置与服务器当前状态矛盾（不可能移动）时必须拒绝"""
        _record_line(self.system, "target", 0.0, 2.0, 0.1,
                     lambda t: (10.0, 0.0, 0.0))
        fake_hit = {"hit": True, "latency_used": 0.1, "rewind_time": 0.9,
                    "target_position": (100.0, 0.0, 0.0)}
        check = self.system.compensation_check("shooter", "target", fake_hit, 1.0)
        assert check["valid"] is False
        assert any("不可能" in r for r in check["reasons"])

    def test_case_14(self):
        """通过伪造 ping 获得的命中必须被反作弊拒绝"""
        report, hit = self._legit_hit()
        forged_report = dict(report, forged_ping=True)
        check = self.system.compensation_check("shooter", "target", hit, 1.0,
                                               latency_report=forged_report)
        assert check["valid"] is False
        assert any("伪造" in r for r in check["reasons"])

    def test_case_15(self):
        """合法命中必须通过全部合理性校验"""
        report, hit = self._legit_hit()
        check = self.system.compensation_check("shooter", "target", hit, 1.0,
                                               latency_report=report)
        assert check["valid"] is True
        assert check["reasons"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
