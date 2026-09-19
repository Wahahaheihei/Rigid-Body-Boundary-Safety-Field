# -*- coding: utf-8 -*-
"""Representative implementation of the rigid-body boundary-source safety-field model.

This module contains the computational and rendering functions required by the
public Scenario 2 example.
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.optimize import lsq_linear

# ---------------------------------------------------------------------
# Baseline configuration used for the public Scenario 2 example
# ---------------------------------------------------------------------
CONFIG = {
    "X_LIM": (-92, 92),
    "Y_LIM": (-28, 28),

    "SHAPE": {
        "FUS_L": 18.0,
        "FUS_W": 2.8,
        "NOSE_SEGMENTS": 14,
        "WING_SPAN": 8.5,
        "WING_ROOT_CHORD": 5.0,
        "WING_TIP_CHORD": 2.4,
        "WING_LE_SWEEP": 4.4,
        "WING_X": 2.0,
        "TAIL_SPAN": 3.0,
        "TAIL_ROOT_CHORD": 2.8,
        "TAIL_TIP_CHORD": 1.2,
        "TAIL_LE_SWEEP": 2.4,
        "TAIL_X": -7.2,
    },

    "TOTAL_BOUNDARY_POINTS": 384,
    "MIN_POINTS_PER_EDGE": 1,

    "MODEL": {
        "K": 0.5,
        "k1": 1.2,
        # Phenomenological speed-scale parameter in the dynamic correction.
        # It is expressed in the same velocity unit as the local boundary
        # velocities used by this implementation (m/s).
        "k3": 45.0,
        "alpha": 0.06,
        "eps": 0.35,
        "d0": 1.6,
    },

    # Normalized aircraft-level total source strength M=1 for each aircraft.
    "TOTAL_WEIGHT": {"A": 1.0, "B": 1.0},

    "SOLVER": {
        "LAMBDA_M": 1.0e6,
        "LAMBDA_L": 3.0e-3,
        "TOL": 1.0e-10,
        "MAX_ITER": 400,
    },

    "RENDER": {
        "GRID_RES": 220,
        "SIGMA": 1.75,
        "BAND_WIDTH": 8.5,
        "BAND_SOFT": 5.0,
        "ALPHA_MAX": 0.95,
        "GAMMA": 0.85,
        "VMAX_PCT": 95.0,
        "CMAP": "turbo",
    },

    "NORMALIZATION": {
        "E_REF_VALUE": 516.4152141159,
        "T_REF": 1.0,
        "REFERENCE_NAME": "Scenario 2 deterministic baseline encounter",
    },

    "PARALLEL_OPPOSITE_SCENE": {
        "LANE_GAP": 21.0,
        "A_START": (-76.0, 10.5),
        "A_END_X": 76.0,
        "A_HEADING_DEG": 0.0,
        "A_SPEED": 5.2,
        "B_START": (76.0, -10.5),
        "B_END_X": -76.0,
        "B_HEADING_DEG": 180.0,
        "B_SPEED": 5.2,
        "CENTER_X": 0.0,
        "TIME_DT": 0.1,
        "POST_STOP_HOLD": 0.0,
    },
}



def deg2rad(d):
    return np.deg2rad(float(d))


def rot2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def perp(v: np.ndarray) -> np.ndarray:
    return np.stack([-v[..., 1], v[..., 0]], axis=-1)


def rigid_points_velocity(c: np.ndarray, v: np.ndarray, omega: float, P: np.ndarray) -> np.ndarray:
    return v[None, :] + omega * perp(P - c[None, :])


def point_in_poly(pt: np.ndarray, poly: np.ndarray) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < xinters:
                inside = not inside
    return inside


def points_inside_union(P: np.ndarray, polys_world: list) -> np.ndarray:
    inside = np.zeros(P.shape[0], dtype=bool)
    for poly in polys_world:
        for i in range(P.shape[0]):
            if not inside[i]:
                inside[i] = point_in_poly(P[i], poly)
    return inside


def min_dist_points_to_polys(P: np.ndarray, polys_world: list) -> np.ndarray:
    M = P.shape[0]
    best_d2 = np.full(M, np.inf, dtype=float)
    for poly in polys_world:
        n = len(poly)
        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]
            ab = b - a
            denom = float(np.dot(ab, ab)) + 1e-12
            ap = P - a
            t = (ap @ ab) / denom
            t = np.clip(t, 0.0, 1.0)
            proj = a + t[:, None] * ab[None, :]
            d2 = np.sum((P - proj) ** 2, axis=1)
            best_d2 = np.minimum(best_d2, d2)
    return np.sqrt(best_d2)


class AircraftShape:
    def __init__(self, cfg):
        L = float(cfg["FUS_L"])
        W = float(cfg["FUS_W"])
        nose_segments = int(cfg.get("NOSE_SEGMENTS", 14))

        x_tail = -L / 2.0
        x_nose_center = L / 2.0 - W / 2.0
        theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, nose_segments + 1)
        nose_arc = np.column_stack([
            x_nose_center + (W / 2.0) * np.cos(theta),
            (W / 2.0) * np.sin(theta),
        ])
        fus_pts = [[x_tail, -W / 2.0], [x_nose_center, -W / 2.0]]
        fus_pts.extend(nose_arc[1:-1].tolist())
        fus_pts.append([x_nose_center, W / 2.0])
        fus_pts.append([x_tail, W / 2.0])
        fus = np.array(fus_pts, dtype=float)

        span = float(cfg["WING_SPAN"])
        root = float(cfg["WING_ROOT_CHORD"])
        tip = float(cfg["WING_TIP_CHORD"])
        le_sweep = float(cfg["WING_LE_SWEEP"])
        wing_x = float(cfg["WING_X"])

        y_root_r = -W / 2.0
        y_tip_r = y_root_r - span
        x_root_le = wing_x
        x_root_te = wing_x - root
        x_tip_le = x_root_le - le_sweep
        x_tip_te = x_tip_le - tip
        right_wing = np.array([
            [x_root_le, y_root_r],
            [x_root_te, y_root_r],
            [x_tip_te, y_tip_r],
            [x_tip_le, y_tip_r],
        ], dtype=float)
        left_wing = right_wing.copy()
        left_wing[:, 1] *= -1.0

        tail_span = float(cfg["TAIL_SPAN"])
        tail_root = float(cfg["TAIL_ROOT_CHORD"])
        tail_tip = float(cfg["TAIL_TIP_CHORD"])
        tail_le_sweep = float(cfg["TAIL_LE_SWEEP"])
        tail_x = float(cfg["TAIL_X"])

        y_tail_root_r = -W / 2.0
        y_tail_tip_r = y_tail_root_r - tail_span
        x_tail_root_le = tail_x
        x_tail_root_te = tail_x - tail_root
        x_tail_tip_le = x_tail_root_le - tail_le_sweep
        x_tail_tip_te = x_tail_tip_le - tail_tip
        right_tail = np.array([
            [x_tail_root_le, y_tail_root_r],
            [x_tail_root_te, y_tail_root_r],
            [x_tail_tip_te, y_tail_tip_r],
            [x_tail_tip_le, y_tail_tip_r],
        ], dtype=float)
        left_tail = right_tail.copy()
        left_tail[:, 1] *= -1.0

        self.components_body = [fus, right_wing, left_wing, right_tail, left_tail]

    def world_polys(self, center: np.ndarray, heading: float):
        R = rot2d(heading)
        return [poly @ R.T + center for poly in self.components_body]

    def edge_list_world(self, center: np.ndarray, heading: float):
        polys_world = self.world_polys(center, heading)
        edges = []
        for poly_w in polys_world:
            n = len(poly_w)
            for i in range(n):
                a = poly_w[i]
                b = poly_w[(i + 1) % n]
                edges.append((a, b))
    def sample_points_world_fixed_total(
        self,
        center: np.ndarray,
        heading: float,
        total_points: int,
        min_points_per_edge: int = 1,
        return_component_slices: bool = False,
    ):
        """
        Sample a fixed total number of ordered boundary points.

        Points are allocated to line segments proportionally to segment length.
        The sampling order is preserved component by component so that an
        adjacency-based smoothness matrix can be assembled without linking
        unrelated components.
        """
        polys_world = self.world_polys(center, heading)

        edges = []
        for comp_id, poly in enumerate(polys_world):
            n = len(poly)
            for i in range(n):
                edges.append((poly[i], poly[(i + 1) % n], comp_id))

        n_edges = len(edges)
        if total_points < n_edges * min_points_per_edge:
            raise ValueError(f"total_points={total_points} 小于最小需求 {n_edges * min_points_per_edge}")

        lengths = np.array([np.linalg.norm(b - a) for a, b, _ in edges], dtype=float)
        total_length = float(np.sum(lengths))
        if total_length <= 1e-12:
            raise ValueError("边界总长度过小，无法采样")

        raw = total_points * lengths / total_length
        counts = np.floor(raw).astype(int)
        counts = np.maximum(counts, min_points_per_edge)

        deficit = int(total_points - np.sum(counts))
        if deficit > 0:
            frac = raw - np.floor(raw)
            order = np.argsort(-frac)
            for k in range(deficit):
                counts[order[k % n_edges]] += 1
        elif deficit < 0:
            reducible = counts - min_points_per_edge
            frac = raw - np.floor(raw)
            order = np.argsort(frac)
            need = -deficit
            for idx in order:
                if need <= 0:
                    break
                take = min(int(reducible[idx]), need)
                if take > 0:
                    counts[idx] -= take
                    need -= take
            if need > 0:
                raise RuntimeError("无法在保持每边最小点数的前提下分配固定总点数")

        pts_by_component = []
        component_slices = []
        start = 0
        for comp_id in range(len(polys_world)):
            comp_parts = []
            for edge_idx, (a, b, cid) in enumerate(edges):
                if cid != comp_id:
                    continue
                cnt = int(counts[edge_idx])
                t = np.linspace(0.0, 1.0, cnt, endpoint=False)
                seg_pts = (1.0 - t)[:, None] * a[None, :] + t[:, None] * b[None, :]
                comp_parts.append(seg_pts)

            comp_pts = np.vstack(comp_parts).astype(float)
            pts_by_component.append(comp_pts)
            component_slices.append(slice(start, start + len(comp_pts)))
            start += len(comp_pts)

        P = np.vstack(pts_by_component).astype(float)
        if P.shape[0] != total_points:
            raise RuntimeError(f"边界采样点数量异常：得到 {P.shape[0]}，预期 {total_points}")

        if return_component_slices:
            return P, component_slices
        return P

        return np.vstack(pts).astype(float)


def build_componentwise_difference_matrix(component_slices: list, n_points: int) -> np.ndarray:
    """Construct the first-order adjacency difference matrix L."""
    rows = []
    for sl in component_slices:
        idx = np.arange(sl.start, sl.stop, dtype=int)
        if len(idx) <= 1:
            continue
        for k, i in enumerate(idx):
            j = idx[(k + 1) % len(idx)]
            row = np.zeros(n_points, dtype=float)
            row[i] = -1.0
            row[j] = +1.0
            rows.append(row)
    if not rows:
        return np.zeros((0, n_points), dtype=float)
    return np.vstack(rows)


def field_strength_vector(rvec: np.ndarray, r: np.ndarray, r_soft: np.ndarray, k1: float) -> np.ndarray:
    e = rvec / (r[:, None] + 1e-12)
    mag = 1.0 / (r_soft ** k1)
    return e * mag[:, None]


def potential_integral_kernel(r_soft: np.ndarray, k1: float) -> np.ndarray:
    """Path-integral kernel used by the current baseline formulation.

    The public baseline uses k1=1.2. The singular special case k1=1 is not used.
    """
    if abs(k1 - 1.0) < 1e-12:
        raise ValueError("k1=1 is not supported by this representative implementation.")
    return 1.0 / ((k1 - 1.0) * (r_soft ** (k1 - 1.0)))


def moving_flag(v_center: np.ndarray, omega: float, tol: float = 1e-12) -> bool:
    return (np.linalg.norm(v_center) > tol) or (abs(omega) > tol)


def solve_masses_equipotential_nonnegative(
    P_all: np.ndarray,
    labels: np.ndarray,
    component_slices: list,
    centers: list,
    v_centers: list,
    omegas: list,
    moving_flags: list,
    M_targets: np.ndarray,
    K: float,
    k1: float,
    k3: float,
    eps: float,
    d0: float,
    lambda_M: float,
    lambda_L: float,
    solver_tol: float,
    max_iter: int,
):
    """
    Solve the constrained virtual-mass inversion:

        min_{m >= 0, c}
            ||A m - S c||_2^2
          + lambda_M ||B m - M||_2^2
          + lambda_L ||L m||_2^2

    where L is the first-order adjacency difference matrix.
    """
    N = P_all.shape[0]
    n_obj = int(np.max(labels) + 1)

    S = np.zeros((N, n_obj), dtype=float)
    S[np.arange(N), labels] = 1.0

    B = np.zeros((n_obj, N), dtype=float)
    for obj in range(n_obj):
        B[obj, labels == obj] = 1.0

    L = build_componentwise_difference_matrix(component_slices, N)

    source_vel_all = np.zeros_like(P_all)
    for obj in range(n_obj):
        idx = (labels == obj)
        source_vel_all[idx] = rigid_points_velocity(centers[obj], v_centers[obj], omegas[obj], P_all[idx])

    rvec = P_all[:, None, :] - P_all[None, :, :]
    r = np.sqrt(np.sum(rvec * rvec, axis=2) + eps * eps)
    r_soft = r + d0
    e = rvec / (r[..., None] + 1e-12)

    kernel = potential_integral_kernel(r_soft, k1)
    gain = np.ones((N, N), dtype=float)
    for obj in range(n_obj):
        idx = np.where(labels == obj)[0]
        if moving_flags[obj]:
            proj = np.einsum('ijd,jd->ij', e[:, idx, :], source_vel_all[idx])
            gain[:, idx] = k3 / np.maximum(k3 - proj, 1e-8)

    A = K * gain * kernel

    blocks = [np.hstack([A, -S])]
    rhs_blocks = [np.zeros(N, dtype=float)]

    if lambda_M < 0.0:
        raise ValueError("lambda_M 必须为非负数")
    if lambda_M > 0.0:
        w_M = math.sqrt(float(lambda_M))
        blocks.append(np.hstack([w_M * B, np.zeros((n_obj, n_obj), dtype=float)]))
        rhs_blocks.append(w_M * M_targets)

    if lambda_L < 0.0:
        raise ValueError("lambda_L 必须为非负数")
    if lambda_L > 0.0 and L.shape[0] > 0:
        w_L = math.sqrt(float(lambda_L))
        blocks.append(np.hstack([w_L * L, np.zeros((L.shape[0], n_obj), dtype=float)]))
        rhs_blocks.append(np.zeros(L.shape[0], dtype=float))

    Msys = np.vstack(blocks)
    rhs = np.concatenate(rhs_blocks)

    lb = np.concatenate([np.zeros(N, dtype=float), np.full(n_obj, -np.inf, dtype=float)])
    ub = np.full(N + n_obj, np.inf, dtype=float)

    res = lsq_linear(
        Msys,
        rhs,
        bounds=(lb, ub),
        method='bvls',
        tol=float(solver_tol),
        max_iter=int(max_iter),
        verbose=0,
    )
    if not res.success:
        raise RuntimeError(f"非负约束求解失败：status={res.status}, message={res.message}")

    sol = res.x
    m = sol[:N]
    C = sol[N:]
    mass_sum = B @ m
    diagnostics = {
        'solver_cost': float(res.cost),
        'solver_optimality': float(res.optimality),
        'solver_iterations': int(res.nit) if res.nit is not None else -1,
        'n_smoothness_rows': int(L.shape[0]),
    }
    return m, C, mass_sum, diagnostics


def compute_external_aspe_and_rate(
    P_all: np.ndarray,
    labels: np.ndarray,
    m: np.ndarray,
    centers: list,
    v_centers: list,
    omegas: list,
    moving_flags: list,
    M_recv_list: np.ndarray,
    K: float,
    k1: float,
    k3: float,
    eps: float,
    d0: float,
):
    N = P_all.shape[0]
    n_obj = int(np.max(labels) + 1)

    source_vel_all = np.zeros_like(P_all)
    for obj in range(n_obj):
        idx = (labels == obj)
        source_vel_all[idx] = rigid_points_velocity(centers[obj], v_centers[obj], omegas[obj], P_all[idx])

    recv_vel_all = np.zeros_like(P_all)
    for obj in range(n_obj):
        idx = (labels == obj)
        recv_vel_all[idx] = rigid_points_velocity(centers[obj], v_centers[obj], omegas[obj], P_all[idx])

    rvec = P_all[:, None, :] - P_all[None, :, :]
    r = np.sqrt(np.sum(rvec * rvec, axis=2) + eps * eps)
    r_soft = r + d0
    e = rvec / (r[..., None] + 1e-12)

    gain = np.ones((N, N), dtype=float)
    for obj in range(n_obj):
        idx = np.where(labels == obj)[0]
        if moving_flags[obj]:
            proj = np.einsum('ijd,jd->ij', e[:, idx, :], source_vel_all[idx])
            gain[:, idx] = k3 / np.maximum(k3 - proj, 1e-8)

    E_src = K * gain[..., None] * field_strength_vector(
        rvec.reshape(-1, 2), r.reshape(-1), r_soft.reshape(-1), k1
    ).reshape(N, N, 2)
    kernel = potential_integral_kernel(r_soft, k1)

    same_obj_mask = labels[:, None] == labels[None, :]
    cross_mask = ~same_obj_mask

    recv_scale = M_recv_list[labels].astype(float)

    aspe_pair = recv_scale[:, None] * K * gain * kernel * m[None, :]
    aspe_pair *= cross_mask
    ASPE = np.sum(aspe_pair, axis=1)

    F = recv_scale[:, None, None] * (m[None, :, None] * E_src)
    dv = source_vel_all[None, :, :] - recv_vel_all[:, None, :]
    dot_pair = np.sum(F * dv, axis=2) * cross_mask
    ASPE_dot = np.sum(dot_pair, axis=1)
    return ASPE, ASPE_dot


def build_parallel_opposite_trajectories(scene_cfg: dict):
    lane_gap = float(scene_cfg['LANE_GAP'])
    a_start = np.array(scene_cfg['A_START'], dtype=float)
    b_start = np.array(scene_cfg['B_START'], dtype=float)
    center_x = float(scene_cfg.get('CENTER_X', 0.0))

    # 自动修正为严格平行且关于 y=0 对称
    a_start[1] = +0.5 * lane_gap
    b_start[1] = -0.5 * lane_gap

    # 让两机在 x = CENTER_X 附近交会
    half_span_A = abs(float(scene_cfg['A_END_X']) - float(a_start[0])) * 0.5
    half_span_B = abs(float(b_start[0]) - float(scene_cfg['B_END_X'])) * 0.5
    span = min(half_span_A, half_span_B)
    a_start[0] = center_x - span
    b_start[0] = center_x + span
    a_end_x = center_x + span
    b_end_x = center_x - span

    a_speed = float(scene_cfg['A_SPEED'])
    b_speed = float(scene_cfg['B_SPEED'])
    if min(a_speed, b_speed) <= 0:
        raise ValueError('A_SPEED 与 B_SPEED 必须为正')

    t_A = abs(a_end_x - a_start[0]) / a_speed
    t_B = abs(b_start[0] - b_end_x) / b_speed

    return {
        'lane_gap': lane_gap,
        'A': {
            'start': a_start,
            'end_x': a_end_x,
            'y': float(a_start[1]),
            'heading': deg2rad(scene_cfg['A_HEADING_DEG']),
            'speed': a_speed,
            'vx': a_speed,
            't_remove': t_A,
        },
        'B': {
            'start': b_start,
            'end_x': b_end_x,
            'y': float(b_start[1]),
            'heading': deg2rad(scene_cfg['B_HEADING_DEG']),
            'speed': b_speed,
            'vx': -b_speed,
            't_remove': t_B,
        },
        't_interaction_end': min(t_A, t_B),
        't_all_removed': max(t_A, t_B),
    }


def state_parallel_aircraft_at_time(t: float, traj: dict, name: str):
    start = np.asarray(traj['start'], dtype=float)
    y = float(traj['y'])
    vx = float(traj['vx'])
    heading = float(traj['heading'])
    t_remove = float(traj['t_remove'])

    if t >= t_remove:
        return {
            'active': False,
            'removed': True,
            'name': name,
            'segment': f'{name}_removed',
            'center': np.array([float(traj['end_x']), y], dtype=float),
            'heading': heading,
            'vel': np.array([0.0, 0.0], dtype=float),
            'acc': np.array([0.0, 0.0], dtype=float),
            'omega': 0.0,
            'weight': float(CONFIG['TOTAL_WEIGHT'][name]),
        }

    if t <= 0.0:
        pos = start.copy()
        vel = np.array([vx, 0.0], dtype=float)
        seg = f'{name}_start'
    else:
        x = start[0] + vx * t
        pos = np.array([x, y], dtype=float)
        vel = np.array([vx, 0.0], dtype=float)
        seg = f'{name}_cruise'

    return {
        'active': True,
        'removed': False,
        'name': name,
        'segment': seg,
        'center': pos,
        'heading': heading,
        'vel': vel,
        'acc': np.array([0.0, 0.0], dtype=float),
        'omega': 0.0,
        'weight': float(CONFIG['TOTAL_WEIGHT'][name]),
    }


def build_scene_objects_at_time(t: float, trajs: dict):
    A_state = state_parallel_aircraft_at_time(t, trajs['A'], 'A')
    B_state = state_parallel_aircraft_at_time(t, trajs['B'], 'B')
    states = {'A': A_state, 'B': B_state}
    objects = [s for s in [A_state, B_state] if s['active']]
    return objects, states


def compute_single_time_step_multi(shape: AircraftShape, objects: list):
    total_points = int(CONFIG['TOTAL_BOUNDARY_POINTS'])
    min_points = int(CONFIG['MIN_POINTS_PER_EDGE'])

    P_list = []
    labels = []
    centers = []
    v_centers = []
    omegas = []
    moving_flags = []
    M_targets = []
    object_slices = []
    component_slices = []

    start = 0
    for idx, obj in enumerate(objects):
        center = np.asarray(obj['center'], dtype=float)
        heading = float(obj['heading'])
        vel = np.asarray(obj['vel'], dtype=float)
        omega = float(obj['omega'])
        Pi, comp_slices_local = shape.sample_points_world_fixed_total(
            center, heading, total_points, min_points, return_component_slices=True
        )
        n = len(Pi)
        P_list.append(Pi)
        labels.extend([idx] * n)
        centers.append(center)
        v_centers.append(vel)
        omegas.append(omega)
        moving_flags.append(moving_flag(vel, omega))
        M_targets.append(float(obj['weight']))
        object_slices.append(slice(start, start + n))
        for sl in comp_slices_local:
            component_slices.append(slice(start + sl.start, start + sl.stop))
        start += n

    P_all = np.vstack(P_list)
    labels = np.array(labels, dtype=int)
    M_targets = np.array(M_targets, dtype=float)

    model = CONFIG['MODEL']
    solver = CONFIG['SOLVER']
    m, C, mass_sum, solve_diag = solve_masses_equipotential_nonnegative(
        P_all=P_all,
        labels=labels,
        component_slices=component_slices,
        centers=centers,
        v_centers=v_centers,
        omegas=omegas,
        moving_flags=moving_flags,
        M_targets=M_targets,
        K=model['K'],
        k1=model['k1'],
        k3=model['k3'],
        eps=model['eps'],
        d0=model['d0'],
        lambda_M=solver['LAMBDA_M'],
        lambda_L=solver['LAMBDA_L'],
        solver_tol=solver['TOL'],
        max_iter=solver['MAX_ITER'],
    )
    ASPE, ASPE_dot = compute_external_aspe_and_rate(
        P_all=P_all,
        labels=labels,
        m=m,
        centers=centers,
        v_centers=v_centers,
        omegas=omegas,
        moving_flags=moving_flags,
        M_recv_list=M_targets,
        K=model['K'],
        k1=model['k1'],
        k3=model['k3'],
        eps=model['eps'],
        d0=model['d0'],
    )
    # Raw aircraft-level ASPE / ASPE-dot are available before E_ref is known.
    total_aspe_by_object = [float(np.sum(ASPE[labels == i])) for i in range(len(objects))]
    total_aspe_rate_by_object = [float(np.sum(ASPE_dot[labels == i])) for i in range(len(objects))]

    # During the first (reference) pass E_REF_VALUE is intentionally None.
    # After E_ref has been determined, selected snapshot states are recomputed
    # with exactly the same model and normalized here for rendering.
    norm_cfg = CONFIG.get('NORMALIZATION', {})
    e_ref = norm_cfg.get('E_REF_VALUE', None)
    t_ref = float(norm_cfg.get('T_REF', 1.0))
    if e_ref is None:
        ASPE_star = None
        ASPE_dot_star = None
        ADSI = None
        total_adsi_by_object = None
    else:
        e_ref = float(e_ref)
        if (not np.isfinite(e_ref)) or e_ref <= 0.0:
            raise ValueError(f'E_ref 必须为正有限值，当前为 {e_ref}')
        ASPE_star = ASPE / e_ref
        ASPE_dot_star = (t_ref / e_ref) * ASPE_dot
        ADSI = model['alpha'] * ASPE_star + (1.0 - model['alpha']) * ASPE_dot_star
        total_adsi_by_object = [float(np.sum(ADSI[labels == i])) for i in range(len(objects))]

    return {
        'P_all': P_all,
        'labels': labels,
        'object_points': P_list,
        'object_slices': object_slices,
        'component_slices': component_slices,
        'centers': centers,
        'v_centers': v_centers,
        'omegas': omegas,
        'moving_flags': moving_flags,
        'M_targets': M_targets,
        'm': m,
        'C': C,
        'mass_sum': mass_sum,
        'solve_diag': solve_diag,
        'ASPE': ASPE,
        'ASPE_dot': ASPE_dot,
        'ASPE_star': ASPE_star,
        'ASPE_dot_star': ASPE_dot_star,
        'ADSI': ADSI,
        'total_aspe_by_object': total_aspe_by_object,
        'total_aspe_rate_by_object': total_aspe_rate_by_object,
        'total_adsi_by_object': total_adsi_by_object,
    }


def draw_aircraft(ax, polys, color, alpha, edge_width=1.0, zorder=3, edge_color='k'):
    for poly in polys:
        ax.add_patch(Polygon(poly, closed=True, facecolor=color, edgecolor=edge_color,
                             alpha=alpha, linewidth=edge_width, zorder=zorder))


def draw_light_band(ax, polys_receiver, X, val_raw, cmap_name, vmin, vmax):
    Rcfg = CONFIG["RENDER"]
    res = int(Rcfg["GRID_RES"])
    sigma = float(Rcfg["SIGMA"])
    band_w = float(Rcfg["BAND_WIDTH"])
    band_soft = float(Rcfg["BAND_SOFT"])
    alpha_max = float(Rcfg["ALPHA_MAX"])
    gamma = float(Rcfg["GAMMA"])
    vmax_pct = float(Rcfg["VMAX_PCT"])

    all_pts = np.vstack(polys_receiver)
    xmin0, ymin0 = np.min(all_pts, axis=0)
    xmax0, ymax0 = np.max(all_pts, axis=0)
    xmin, ymin = xmin0 - band_w, ymin0 - band_w
    xmax, ymax = xmax0 + band_w, ymax0 + band_w

    xs = np.linspace(xmin, xmax, res)
    ys = np.linspace(ymin, ymax, res)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    P = np.stack([XX.ravel(), YY.ravel()], axis=1)

    dist = min_dist_points_to_polys(P, polys_receiver)
    idx0 = np.where(dist <= band_w)[0]
    if idx0.size == 0:
        return None

    P0 = P[idx0]
    inside = points_inside_union(P0, polys_receiver)
    idx = idx0[~inside]
    if idx.size == 0:
        return None

    Pm = P[idx]

    fade = np.zeros(P.shape[0], dtype=float)
    fade[idx] = np.exp(- (dist[idx] / max(band_soft, 1e-6)) ** 2)

    s2 = 2.0 * (sigma ** 2)
    S = np.zeros(idx.size, dtype=float)
    V = np.zeros(idx.size, dtype=float)

    for xp, vp in zip(X, val_raw):
        dx = Pm[:, 0] - xp[0]
        dy = Pm[:, 1] - xp[1]
        g = np.exp(-(dx * dx + dy * dy) / s2)
        S += g
        V += vp * g

    C = V / (S + 1e-12)

    Spos = S[S > 0]
    if Spos.size >= 10:
        s_ref = float(np.percentile(Spos, vmax_pct))
        s_ref = max(s_ref, 1e-12)
    else:
        s_ref = float(np.max(S) + 1e-12)

    S_norm = np.clip(S / s_ref, 0.0, 1.0) ** gamma

    C_full = np.zeros(P.shape[0], dtype=float)
    A_full = np.zeros(P.shape[0], dtype=float)
    C_full[idx] = C
    A_full[idx] = np.clip(alpha_max * S_norm * fade[idx], 0.0, 1.0)

    C_img = C_full.reshape((res, res))
    A_img = A_full.reshape((res, res))

    im = ax.imshow(
        C_img,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        interpolation="bilinear",
        cmap=cmap_name,
        vmin=vmin,
        vmax=vmax,
        alpha=A_img,
        zorder=2,
    )
    return im


def draw_velocity_arrow(ax, center, vel, color, scale=5.2, zorder=8):
    vel = np.asarray(vel, dtype=float)
    vnorm = float(np.linalg.norm(vel))
    if vnorm <= 1e-8:
        return
    vdir = vel / vnorm
    start = np.asarray(center, dtype=float) - 0.15 * scale * vdir
    ax.annotate(
        '',
        xy=(start[0] + scale * vdir[0], start[1] + scale * vdir[1]),
        xytext=(start[0], start[1]),
        arrowprops=dict(arrowstyle='-|>', lw=1.8, color=color, shrinkA=0, shrinkB=0, mutation_scale=14),
        zorder=zorder,
    )


def draw_parallel_route_guides(ax, trajs: dict):
    x_left = min(trajs['A']['start'][0], trajs['B']['end_x']) - 8.0
    x_right = max(trajs['A']['end_x'], trajs['B']['start'][0]) + 8.0
    yA = float(trajs['A']['y'])
    yB = float(trajs['B']['y'])
    y_mid = 0.5 * (yA + yB)

    ax.plot([x_left, x_right], [yA, yA], linestyle='--', linewidth=1.3, color='gray', alpha=0.72, zorder=0)
    ax.plot([x_left, x_right], [yB, yB], linestyle='--', linewidth=1.3, color='gray', alpha=0.72, zorder=0)
    ax.plot([0.0, 0.0], [yB - 4.0, yA + 4.0], linestyle=':', linewidth=1.0, color='gray', alpha=0.65, zorder=0)

    ax.text(x_right - 2.0, yA + 0.9, 'Route A  ->', fontsize=8.5, ha='right', va='bottom', color='dimgray')
    ax.text(x_left + 2.0, yB + 0.9, '<-  Route B', fontsize=8.5, ha='left', va='bottom', color='dimgray')
    ax.text(0.0, y_mid, 'meeting zone', fontsize=8.3, ha='center', va='center', color='dimgray',
            bbox=dict(boxstyle='round,pad=0.16', fc='white', alpha=0.72, ec='none'))
