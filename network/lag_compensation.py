"""
延迟补偿系统 - 核心模块（修复版）

修复内容：
- 世界回滚包含姿态（站/蹲/跳）与头部位置
- 延迟计算综合 ping、网络抖动、服务器处理耗时，且有上限
- 命中检测考虑子弹飞行时间、距离与目标移动速度
- 历史记录采样率自适应（速度越快采样越密），插值准确
- 延迟补偿有合理性校验与反作弊（伪造 ping 检测、不可能命中拒绝）
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import bisect
import math
import time

Vector3 = Tuple[float, float, float]

# 各姿态下头部相对脚底的高度偏移
POSTURE_HEAD_OFFSET: Dict[str, float] = {
    "standing": 1.7,
    "crouching": 1.1,
    "jumping": 1.7,
}

# 各姿态下碰撞体高度
POSTURE_BODY_HEIGHT: Dict[str, float] = {
    "standing": 1.8,
    "crouching": 1.2,
    "jumping": 1.8,
}

DEFAULT_CONFIG = {
    "max_latency": 0.25,            # 延迟补偿上限（秒），防止高延迟玩家获得超大回滚窗口
    "history_duration": 2.0,        # 历史记录保留时长（秒），需覆盖 max_latency
    "base_sample_interval": 0.05,   # 静止时的基础采样间隔（20Hz）
    "min_sample_interval": 1.0 / 120.0,  # 高速移动时的最密采样间隔
    "max_sample_interval": 0.1,     # 最疏采样间隔
    "hitbox_radius": 0.4,           # 身体碰撞半径（米），恒定，不随速度拉伸
    "head_hitbox_radius": 0.2,      # 头部碰撞半径（米）
    "max_bullet_range": 300.0,      # 子弹最大射程（米）
    "max_bullet_flight_time": 1.0,  # 子弹最大飞行时间（秒）
    "max_player_speed": 10.0,       # 玩家合理最大移动速度（米/秒）
    "ping_forgery_tolerance": 0.5,  # 客户端上报 ping 相对服务器实测的容忍比例
    "movement_slack": 1.5,          # 合理性校验的移动容差系数
}


def _add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a: Vector3, s: float) -> Vector3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _length(a: Vector3) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _distance(a: Vector3, b: Vector3) -> float:
    return _length(_sub(a, b))


def _normalize(a: Vector3) -> Vector3:
    n = _length(a)
    if n <= 0.0:
        raise ValueError("无法归一化零向量")
    return (a[0] / n, a[1] / n, a[2] / n)


def _lerp(a: Vector3, b: Vector3, t: float) -> Vector3:
    return (a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t)


@dataclass
class PlayerState:
    """某一时刻的玩家状态快照（位置 + 姿态 + 速度）"""
    player_id: str
    position: Vector3          # 脚底位置
    posture: str = "standing"  # standing / crouching / jumping
    velocity: Vector3 = (0.0, 0.0, 0.0)
    timestamp: float = 0.0

    @property
    def head_position(self) -> Vector3:
        """头部位置，随姿态变化（蹲下时头部更低）"""
        offset = POSTURE_HEAD_OFFSET.get(self.posture, POSTURE_HEAD_OFFSET["standing"])
        return (self.position[0], self.position[1] + offset, self.position[2])

    @property
    def body_center(self) -> Vector3:
        """碰撞体中心，随姿态变化"""
        height = POSTURE_BODY_HEIGHT.get(self.posture, POSTURE_BODY_HEIGHT["standing"])
        return (self.position[0], self.position[1] + height / 2.0, self.position[2])

    @property
    def speed(self) -> float:
        return _length(self.velocity)


class LagCompensation:
    """延迟补偿系统核心类"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = dict(DEFAULT_CONFIG)
        if config:
            self.config.update(config)
        # player_id -> 按时间升序的 PlayerState 列表
        self._history: Dict[str, List[PlayerState]] = {}
        # player_id -> 最近上报的客户端 ping 列表（用于计算抖动）
        self._ping_samples: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # 基础接口
    # ------------------------------------------------------------------
    def update(self, dt: float):
        """每帧更新：清理超出保留时长的历史记录"""
        for player_id in list(self._history):
            self._prune_history(player_id)

    def reset(self):
        """重置状态"""
        self._history = {}
        self._ping_samples = {}

    # ------------------------------------------------------------------
    # 历史记录（自适应采样）
    # ------------------------------------------------------------------
    def _sample_interval(self, speed: float) -> float:
        """根据移动速度自适应采样间隔：速度越快，采样越密"""
        base = self.config["base_sample_interval"]
        min_iv = self.config["min_sample_interval"]
        max_iv = self.config["max_sample_interval"]
        max_speed = max(self.config["max_player_speed"], 1e-6)
        factor = 1.0 + 5.0 * min(speed / max_speed, 1.0)
        return max(min_iv, min(max_iv, base / factor))

    def record_state(self, player_id: str, position: Vector3,
                     posture: str = "standing",
                     velocity: Vector3 = (0.0, 0.0, 0.0),
                     timestamp: Optional[float] = None) -> bool:
        """记录玩家状态。采样间隔随速度自适应；姿态变化必定记录。

        返回 True 表示写入了新样本，False 表示被采样率限流跳过。
        """
        if posture not in POSTURE_HEAD_OFFSET:
            raise ValueError(f"未知姿态: {posture}")
        if timestamp is None:
            timestamp = time.time()
        state = PlayerState(player_id=player_id, position=tuple(position),
                            posture=posture, velocity=tuple(velocity),
                            timestamp=timestamp)
        history = self._history.setdefault(player_id, [])
        if history:
            last = history[-1]
            if timestamp < last.timestamp:
                raise ValueError("历史样本时间戳必须单调递增")
            interval = self._sample_interval(state.speed)
            posture_changed = posture != last.posture
            if not posture_changed and (timestamp - last.timestamp) < interval:
                return False
        history.append(state)
        self._prune_history(player_id)
        return True

    def _prune_history(self, player_id: str):
        history = self._history.get(player_id)
        if not history:
            return
        cutoff = history[-1].timestamp - self.config["history_duration"]
        idx = 0
        while idx < len(history) and history[idx].timestamp < cutoff:
            idx += 1
        # 至少保留两个样本以支持插值
        keep_from = max(0, min(idx, len(history) - 2))
        if keep_from > 0:
            del history[:keep_from]

    def get_state_at(self, player_id: str, timestamp: float) -> Optional[PlayerState]:
        """获取指定时刻的玩家状态，样本间做线性插值（位置/速度），
        姿态取插值区间起始样本的姿态。超出范围时钳制到最近样本。"""
        history = self._history.get(player_id)
        if not history:
            return None
        if timestamp <= history[0].timestamp:
            return history[0]
        if timestamp >= history[-1].timestamp:
            return history[-1]
        times = [s.timestamp for s in history]
        idx = bisect.bisect_right(times, timestamp) - 1
        a, b = history[idx], history[idx + 1]
        span = b.timestamp - a.timestamp
        t = (timestamp - a.timestamp) / span if span > 0 else 0.0
        return PlayerState(
            player_id=player_id,
            position=_lerp(a.position, b.position, t),
            posture=a.posture,
            velocity=_lerp(a.velocity, b.velocity, t),
            timestamp=timestamp,
        )

    def history_length(self, player_id: str) -> int:
        """当前保留的历史样本数（用于验证自适应采样）"""
        return len(self._history.get(player_id, []))

    # ------------------------------------------------------------------
    # 世界回滚（位置 + 姿态 + 头部）
    # ------------------------------------------------------------------
    def rewind_world(self, player_ids: List[str], timestamp: float
                     ) -> Dict[str, PlayerState]:
        """把指定玩家回滚到 timestamp 时刻，返回包含姿态与头部位置的完整状态"""
        rewound: Dict[str, PlayerState] = {}
        for player_id in player_ids:
            state = self.get_state_at(player_id, timestamp)
            if state is not None:
                rewound[player_id] = state
        return rewound

    # ------------------------------------------------------------------
    # 延迟计算（ping + 抖动 + 服务器处理耗时，有上限，防伪造）
    # ------------------------------------------------------------------
    def report_ping(self, player_id: str, client_ping: float):
        """记录一次客户端上报的 ping，用于抖动估计"""
        samples = self._ping_samples.setdefault(player_id, [])
        samples.append(client_ping)
        del samples[:-16]

    def _estimate_jitter(self, player_id: str) -> float:
        """用最近的 ping 样本估计网络抖动（平均绝对偏差）"""
        samples = self._ping_samples.get(player_id, [])
        if len(samples) < 2:
            return 0.0
        mean = sum(samples) / len(samples)
        return sum(abs(s - mean) for s in samples) / len(samples)

    def calculate_latency(self, player_id: str, client_ping: float,
                          server_processing_time: float = 0.005,
                          server_measured_ping: Optional[float] = None,
                          jitter: Optional[float] = None) -> Dict:
        """综合计算补偿延迟。

        - 延迟 = 有效 ping + 抖动 + 服务器处理耗时
        - 客户端上报 ping 显著高于服务器实测时判定为伪造，按实测值计算
        - 最终延迟被钳制到 max_latency，杜绝超大回滚窗口
        """
        if jitter is None:
            jitter = self._estimate_jitter(player_id)
        forged = False
        effective_ping = client_ping
        if server_measured_ping is not None:
            tolerance = self.config["ping_forgery_tolerance"]
            if client_ping > server_measured_ping * (1.0 + tolerance) + 0.01:
                forged = True
                effective_ping = server_measured_ping
        raw = effective_ping + jitter + server_processing_time
        max_latency = self.config["max_latency"]
        latency = min(raw, max_latency)
        return {
            "latency": latency,
            "raw_latency": raw,
            "capped": raw > max_latency,
            "jitter": jitter,
            "server_processing_time": server_processing_time,
            "effective_ping": effective_ping,
            "forged_ping": forged,
        }

    # ------------------------------------------------------------------
    # 命中注册（考虑子弹飞行时间与目标移动）
    # ------------------------------------------------------------------
    def hit_registration(self, shooter_id: str, target_id: str,
                         shot_origin: Vector3, shot_direction: Vector3,
                         bullet_speed: float, fire_time: float,
                         shooter_latency: float = 0.0) -> Dict:
        """判定一次射击是否命中。

        - 按射手延迟把目标回滚到射手视角时刻
        - 迭代求解子弹飞行时间，目标在飞行期间继续按历史轨迹移动
        - 碰撞半径恒定，不随目标速度拉伸
        """
        result = {
            "hit": False,
            "headshot": False,
            "flight_time": 0.0,
            "distance": 0.0,
            "rewind_time": fire_time - shooter_latency,
            "latency_used": shooter_latency,
            "target_position": None,
            "reason": None,
        }
        if bullet_speed <= 0.0:
            result["reason"] = "子弹速度必须为正"
            return result
        direction = _normalize(shot_direction)
        rewind_time = result["rewind_time"]
        if self.get_state_at(target_id, rewind_time) is None:
            result["reason"] = "目标在该时刻无历史记录"
            return result

        def target_center_at(t: float) -> Vector3:
            return self.get_state_at(target_id, t).body_center

        # 迭代求解飞行时间：子弹飞向的是飞行结束后目标所在的位置
        flight_time = _distance(shot_origin, target_center_at(rewind_time)) / bullet_speed
        for _ in range(4):
            aim_point = target_center_at(rewind_time + flight_time)
            flight_time = _distance(shot_origin, aim_point) / bullet_speed

        target_state = self.get_state_at(target_id, rewind_time + flight_time)
        target_center = target_state.body_center
        impact_point = _add(shot_origin, _scale(direction, bullet_speed * flight_time))
        miss_distance = _distance(impact_point, target_center)
        travel_distance = bullet_speed * flight_time

        result["flight_time"] = flight_time
        result["distance"] = travel_distance
        result["target_position"] = target_state.position

        if flight_time > self.config["max_bullet_flight_time"]:
            result["reason"] = "子弹飞行时间超限"
            return result
        if travel_distance > self.config["max_bullet_range"]:
            result["reason"] = "超出子弹最大射程"
            return result

        head_miss = _distance(impact_point, target_state.head_position)
        if head_miss <= self.config["head_hitbox_radius"]:
            result["hit"] = True
            result["headshot"] = True
            return result
        if miss_distance <= self.config["hitbox_radius"]:
            result["hit"] = True
            return result
        result["reason"] = "未命中碰撞盒"
        return result

    # ------------------------------------------------------------------
    # 补偿合理性校验（反作弊）
    # ------------------------------------------------------------------
    def compensation_check(self, shooter_id: str, target_id: str,
                           hit_result: Dict, current_time: float,
                           latency_report: Optional[Dict] = None) -> Dict:
        """校验一次延迟补偿命中是否合理。

        - 拒绝伪造 ping 获得的回滚窗口
        - 拒绝超过延迟上限的补偿
        - 回滚命中的位置必须与服务器当前状态可达（不可能移动则拒绝）
        """
        reasons: List[str] = []
        if latency_report is not None and latency_report.get("forged_ping"):
            reasons.append("检测到伪造 ping")
        latency_used = hit_result.get("latency_used", 0.0)
        if latency_used > self.config["max_latency"] + 1e-9:
            reasons.append("补偿延迟超过上限")
        if not hit_result.get("hit"):
            reasons.append("原始判定未命中")

        rewind_time = hit_result.get("rewind_time")
        target_position = hit_result.get("target_position")
        if rewind_time is not None and target_position is not None:
            current_state = self.get_state_at(target_id, current_time)
            if current_state is not None:
                elapsed = max(0.0, current_time - rewind_time)
                max_displacement = (self.config["max_player_speed"] * elapsed
                                    * self.config["movement_slack"] + 0.5)
                displacement = _distance(target_position, current_state.position)
                if displacement > max_displacement:
                    reasons.append("命中位置与服务器当前状态矛盾（不可能的移动）")

        return {"valid": not reasons, "reasons": reasons}
