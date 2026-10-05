"""
延迟补偿系统 - 核心模块

已修复的问题：
- 世界回滚现在包含玩家姿态（站/蹲/跳）与头部位置，爆头判定基于回滚时刻的真实姿态
- 延迟计算综合客户端 ping、服务器测量 ping、网络抖动与服务器处理耗时，并有上限
- 命中检测考虑子弹飞行时间、飞行距离与目标移动速度（扫掠碰撞，而非静态碰撞盒）
- 历史记录采用自适应采样率（移动越快采样越密），并使用线性插值
- 延迟补偿带合理性校验与 ping 伪造检测（反作弊）
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import time

Vec3 = Tuple[float, float, float]

# 各姿态下的碰撞盒参数：头部高度 / 躯干中心高度 / 躯干半径
POSTURE_HITBOX = {
    "standing": {"head_height": 1.7, "center_height": 0.9, "body_radius": 0.45},
    "crouching": {"head_height": 1.0, "center_height": 0.5, "body_radius": 0.45},
    "jumping": {"head_height": 1.7, "center_height": 0.9, "body_radius": 0.45},
}
DEFAULT_HITBOX = POSTURE_HITBOX["standing"]
HEAD_RADIUS = 0.25

DEFAULT_CONFIG = {
    "max_compensation_ms": 250.0,    # 延迟补偿上限，超过部分不予补偿
    "history_duration": 2.0,         # 历史记录保留时长（秒）
    "base_sample_interval": 0.05,    # 基础采样间隔（20Hz）
    "min_sample_interval": 1.0 / 60.0,  # 自适应采样最小间隔（60Hz）
    "adaptive_ref_speed": 3.0,       # 自适应采样参考速度（m/s）
    "spoof_tolerance_ms": 30.0,      # 上报 ping 与实测 ping 的容忍偏差
    "ping_window": 20,               # ping 测量滑动窗口
    "processing_window": 60,         # 服务器处理耗时滑动窗口
    "max_bullet_distance": 1000.0,   # 子弹最大有效射程（米）
    "plausibility_margin": 1.0,      # 合理性校验的容差（米）
}


@dataclass
class PlayerState:
    """某一时刻的玩家状态快照"""
    player_id: str
    position: Vec3                       # 脚部位置
    posture: str = "standing"            # standing / crouching / jumping
    velocity: Vec3 = (0.0, 0.0, 0.0)
    timestamp: float = 0.0

    @property
    def hitbox(self) -> Dict[str, float]:
        return POSTURE_HITBOX.get(self.posture, DEFAULT_HITBOX)

    @property
    def head_position(self) -> Vec3:
        """头部位置由姿态决定：蹲姿头部明显低于站姿"""
        return (self.position[0],
                self.position[1] + self.hitbox["head_height"],
                self.position[2])

    @property
    def body_center(self) -> Vec3:
        return (self.position[0],
                self.position[1] + self.hitbox["center_height"],
                self.position[2])


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length(a: Vec3) -> float:
    return math.sqrt(_dot(a, a))


def _normalize(a: Vec3) -> Vec3:
    n = _length(a)
    if n <= 1e-12:
        raise ValueError("无法归一化零向量")
    return (a[0] / n, a[1] / n, a[2] / n)


def _lerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    return (a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t)


def _sweep_sphere(origin: Vec3, direction: Vec3, speed: float,
                  center: Vec3, velocity: Vec3, radius: float) -> Optional[float]:
    """求解子弹（匀速直线）与匀速移动球体的最早命中时刻。

    子弹位置 B(t) = origin + direction * speed * t
    球心位置 C(t) = center + velocity * t
    解 |B(t) - C(t)| = radius 的最小非负根；子弹初始即在球内则返回 0。
    """
    r0 = _sub(origin, center)
    r1 = _sub(_mul(direction, speed), velocity)
    c = _dot(r0, r0) - radius * radius
    if c <= 0.0:
        return 0.0
    a = _dot(r1, r1)
    if a <= 1e-12:
        return None
    b = 2.0 * _dot(r0, r1)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    sq = math.sqrt(disc)
    for t in ((-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)):
        if t >= 0.0:
            return t
    return None


class LagCompensation:
    """延迟补偿系统核心类"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = dict(DEFAULT_CONFIG)
        if config:
            self.config.update(config)
        self._history: Dict[str, deque] = {}
        self._current: Dict[str, PlayerState] = {}
        self._pings: Dict[str, deque] = {}
        self._processing: deque = deque(maxlen=int(self.config["processing_window"]))
        self._flagged: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # 状态记录（自适应采样率）
    # ------------------------------------------------------------------
    def sample_interval_for(self, speed: float) -> float:
        """自适应采样间隔：移动越快，采样越密，保证插值精度"""
        base = self.config["base_sample_interval"]
        ref = self.config["adaptive_ref_speed"]
        interval = base * ref / max(speed, ref)
        return max(self.config["min_sample_interval"], min(base, interval))

    def record_state(self, player_id: str, position: Vec3,
                     posture: str = "standing",
                     velocity: Vec3 = (0.0, 0.0, 0.0),
                     timestamp: Optional[float] = None) -> bool:
        """记录玩家状态。返回是否写入了历史样本（自适应采样可能跳过）。"""
        if timestamp is None:
            timestamp = time.time()
        state = PlayerState(player_id=player_id, position=tuple(position),
                            posture=posture, velocity=tuple(velocity),
                            timestamp=timestamp)
        self._current[player_id] = state
        history = self._history.setdefault(player_id, deque())

        interval = self.sample_interval_for(_length(state.velocity))
        sampled = False
        if not history:
            sampled = True
        else:
            last = history[-1]
            # 姿态变化必须立即采样（影响爆头判定），否则按自适应间隔采样
            if last.posture != posture or timestamp - last.timestamp >= interval - 1e-9:
                sampled = True
        if sampled:
            history.append(state)
            horizon = timestamp - self.config["history_duration"]
            while len(history) > 1 and history[0].timestamp < horizon:
                history.popleft()
        return sampled

    # ------------------------------------------------------------------
    # 延迟计算（多因素 + 上限 + 反伪造）
    # ------------------------------------------------------------------
    def record_ping(self, player_id: str, rtt_ms: float):
        """记录服务器实测 ping（RTT，毫秒）"""
        window = self._pings.setdefault(
            player_id, deque(maxlen=int(self.config["ping_window"])))
        window.append(float(rtt_ms))

    def update(self, dt: float):
        """每帧更新，累计服务器处理耗时（秒）"""
        self._processing.append(dt * 1000.0)

    def calculate_latency(self, player_id: str,
                          reported_ping_ms: Optional[float] = None) -> float:
        """计算有效补偿延迟（毫秒）。

        综合：单向网络延迟（ping/2）+ 网络抖动 + 服务器处理耗时，
        并限制在 max_compensation_ms 以内。
        上报 ping 与服务器实测偏差过大时判定为伪造，改用实测值并标记。
        """
        measured = list(self._pings.get(player_id, ()))
        measured_avg = sum(measured) / len(measured) if measured else None

        ping = reported_ping_ms if reported_ping_ms is not None else measured_avg
        if ping is None:
            ping = 0.0
        if (reported_ping_ms is not None and measured_avg is not None
                and abs(reported_ping_ms - measured_avg) > self.config["spoof_tolerance_ms"]):
            self._flagged[player_id] = "ping_spoofing"
            ping = measured_avg

        jitter = (max(measured) - min(measured)) / 2.0 if len(measured) >= 2 else 0.0
        processing = (sum(self._processing) / len(self._processing)
                      if self._processing else 0.0)

        latency = ping / 2.0 + jitter + processing
        return max(0.0, min(latency, self.config["max_compensation_ms"]))

    # ------------------------------------------------------------------
    # 世界回滚（位置 + 姿态 + 头部）
    # ------------------------------------------------------------------
    def get_rewound_state(self, player_id: str, timestamp: float) -> Optional[PlayerState]:
        """获取指定时刻的插值状态；超出历史范围返回 None"""
        history = self._history.get(player_id)
        if not history:
            return None
        samples = list(history)
        if timestamp < samples[0].timestamp - 1e-9:
            return None
        if timestamp >= samples[-1].timestamp:
            last = samples[-1]
            return PlayerState(player_id=player_id, position=last.position,
                               posture=last.posture, velocity=last.velocity,
                               timestamp=timestamp)
        for i in range(len(samples) - 1):
            a, b = samples[i], samples[i + 1]
            if a.timestamp - 1e-9 <= timestamp <= b.timestamp + 1e-9:
                span = b.timestamp - a.timestamp
                t = 0.0 if span <= 1e-12 else (timestamp - a.timestamp) / span
                return PlayerState(
                    player_id=player_id,
                    position=_lerp(a.position, b.position, t),
                    posture=a.posture,  # 姿态取区间起点（阶跃变化，不做插值）
                    velocity=_lerp(a.velocity, b.velocity, t),
                    timestamp=timestamp)
        return None

    def rewind_world(self, rewind_ms: float,
                     now: Optional[float] = None) -> Dict[str, PlayerState]:
        """把世界回滚 rewind_ms 毫秒，返回所有玩家当时的状态（含姿态与头部）"""
        if now is None:
            now = max((s.timestamp for s in self._current.values()), default=time.time())
        target = now - rewind_ms / 1000.0
        rewound = {}
        for player_id in self._current:
            state = self.get_rewound_state(player_id, target)
            if state is not None:
                rewound[player_id] = state
        return rewound

    # ------------------------------------------------------------------
    # 命中注册（考虑子弹飞行时间、距离与目标移动速度）
    # ------------------------------------------------------------------
    def hit_registration(self, shooter_id: str, target_id: str,
                         origin: Vec3, direction: Vec3, bullet_speed: float,
                         shot_time: Optional[float] = None,
                         reported_ping_ms: Optional[float] = None,
                         max_distance: Optional[float] = None) -> Dict:
        """判定一次射击是否命中。

        流程：计算射手有效延迟 -> 将目标回滚到开枪时刻 -> 以有限弹速
        对移动目标做扫掠碰撞检测 -> 校验子弹飞行距离。
        """
        miss = {"hit": False, "headshot": False, "flight_time": None,
                "distance": None, "hit_position": None, "reason": None}
        if bullet_speed <= 0:
            miss["reason"] = "invalid_bullet_speed"
            return miss
        direction = _normalize(tuple(direction))
        origin = tuple(origin)
        max_distance = max_distance if max_distance is not None \
            else self.config["max_bullet_distance"]

        latency_ms = self.calculate_latency(shooter_id, reported_ping_ms)
        history = self._history.get(target_id)
        if shot_time is None:
            shot_time = history[-1].timestamp if history else time.time()
        target_time = shot_time - latency_ms / 1000.0
        state = self.get_rewound_state(target_id, target_time)
        if state is None:
            miss["reason"] = "no_history"
            miss["rewind_ms"] = latency_ms
            return miss

        hitbox = state.hitbox
        head_t = _sweep_sphere(origin, direction, bullet_speed,
                               state.head_position, state.velocity, HEAD_RADIUS)
        body_t = _sweep_sphere(origin, direction, bullet_speed,
                               state.body_center, state.velocity,
                               hitbox["body_radius"])

        candidates = [(t, True) for t in (head_t,) if t is not None]
        candidates += [(t, False) for t in (body_t,) if t is not None]
        if not candidates:
            miss["reason"] = "missed"
            miss["rewind_ms"] = latency_ms
            return miss
        flight_time, headshot = min(candidates, key=lambda c: c[0])
        distance = bullet_speed * flight_time
        if distance > max_distance:
            miss["reason"] = "out_of_range"
            miss["flight_time"] = flight_time
            miss["distance"] = distance
            miss["rewind_ms"] = latency_ms
            return miss

        return {
            "hit": True,
            "headshot": headshot,
            "flight_time": flight_time,
            "distance": distance,
            "hit_position": _add(origin, _mul(direction, distance)),
            "rewind_ms": latency_ms,
            "reason": None,
        }

    # ------------------------------------------------------------------
    # 补偿合理性校验（反作弊）
    # ------------------------------------------------------------------
    def compensation_check(self, shooter_id: str, target_id: str,
                           rewind_ms: float,
                           hit_position: Optional[Vec3] = None,
                           reported_ping_ms: Optional[float] = None) -> Dict:
        """校验一次延迟补偿命中是否合理。

        - 回滚窗口不得超过补偿上限
        - 上报 ping 与服务器实测严重不符视为伪造，直接拒绝
        - 命中位置必须与目标当前服务器状态 plausible（偏差不得超过
          目标速度 * 回滚时间 + 容差），防止利用回滚窗口打"瞬移"命中
        """
        reasons = []
        if rewind_ms > self.config["max_compensation_ms"] + 1e-9:
            reasons.append("rewind_exceeds_cap")
        if rewind_ms < 0:
            reasons.append("negative_rewind")

        if shooter_id in self._flagged:
            reasons.append(self._flagged[shooter_id])
        elif reported_ping_ms is not None:
            measured = list(self._pings.get(shooter_id, ()))
            if measured:
                measured_avg = sum(measured) / len(measured)
                if abs(reported_ping_ms - measured_avg) > self.config["spoof_tolerance_ms"]:
                    self._flagged[shooter_id] = "ping_spoofing"
                    reasons.append("ping_spoofing")

        current = self._current.get(target_id)
        if current is not None and hit_position is not None:
            speed = _length(current.velocity)
            max_deviation = (speed * max(rewind_ms, 0.0) / 1000.0
                             + self.config["plausibility_margin"]
                             + current.hitbox["head_height"])
            gap = min(_length(_sub(tuple(hit_position), current.head_position)),
                      _length(_sub(tuple(hit_position), current.body_center)))
            if gap > max_deviation:
                reasons.append("implausible_hit")

        return {"accepted": not reasons, "reasons": reasons}

    # ------------------------------------------------------------------
    def reset(self):
        """重置状态"""
        self._history = {}
        self._current = {}
        self._pings = {}
        self._processing = deque(maxlen=int(self.config["processing_window"]))
        self._flagged = {}
