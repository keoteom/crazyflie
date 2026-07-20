#!/usr/bin/env python3
"""
OpEn NMPC Solver Generator — Crazyflie.
============================================================================
  * Dynamic spherical obstacle : SOFT penalty in the cost  (unchanged).
  * SFC half-space corridor     : HARD inequality via Augmented Lagrangian
                                  (ALM)  ── RE-ADDED in this version.

This re-adds the SFC corridor as a HARD constraint
    n_jm . pos_k - b_jm <= 0   (plane m, horizon step j)
enforced by OpEn's ALM, while keeping the moving obstacle as a SOFT cost
penalty.

WHY soft-obstacle + hard-SFC (not the old hard+hard):
  The old version made BOTH hard and diverged when a moving obstacle entered
  the corridor (jointly infeasible -> ALM never converged). Keeping the
  obstacle soft means that if the obstacle clips the corridor, the obstacle
  penalty yields and the SFC stays feasible. Strictly safer for convergence.

CAVEAT (output side): the C++ node publishes cmd_full_state and the onboard
PID re-tracks a carrot. The SFC bounds the SOLVER's predicted trajectory, NOT
the executed command. This is a guarantee on the plan, not the airframe.

Parameter vector layout (total = 108 + 4*M_MAX*N = 1548):
  [0:8]      x0 = [px,py,pz, vx,vy,vz, phi,theta]
  [8:11]     u_prev = [T, phi_ref, theta_ref]_{k-1}
  [11:101]   p_ref  = 30 reference positions (3*30 = 90)
  [101:104]  p_obs
  [104:107]  v_obs
  [107]      r_obs
  [108:1548] SFC planes. For step j, plane m at  108 + (j*M_MAX + m)*4
             = [n_x, n_y, n_z, b]. Inactive planes zero-padded
             (n=0,b=0 -> g = 0 - 0 = 0 <= 0, trivially satisfied).
"""

import casadi as cs
import opengen as og
import os

# ---------------------------------------------------------------------------
#  Horizon and dynamics (must match nmpc_node.hpp)
# ---------------------------------------------------------------------------
N  = 30
Ts = 0.05
nx = 8
nu = 3

# SFC: per-step plane budget. Pad with zeros if fewer planes are active.
M_MAX = 12

tau_phi   = 0.5
tau_theta = 0.5
K_phi     = 1.0
K_theta   = 1.0
Ax, Ay, Az = 0.1, 0.1, 0.2
g = 9.81

# ---------------------------------------------------------------------------
#  Cost weights
# ---------------------------------------------------------------------------
Qx  = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 8.0, 8.0]   # vel weight 3.0 -> 1.0 (less lag)
Qu  = [5.0, 10.0, 10.0]
QdU = [5.0, 12.0, 12.0]

# Soft-obstacle penalty weight (the obstacle stays SOFT).
W_obs = 1000.0

u_ref_vals = [g, 0.0, 0.0]

u_min = [5.0, -0.35, -0.35]
u_max = [13.5, 0.35, 0.35]

# Obstacle softening radius ramp (must match R_S_MAX in nmpc_node.hpp).
R_S_MAX = 0.1

# ---------------------------------------------------------------------------
#  Parameter layout
# ---------------------------------------------------------------------------
SFC_START = 108
SFC_BLOCK = 4 * M_MAX * N            # 4 * 12 * 30 = 1440
N_PARAMS  = SFC_START + SFC_BLOCK    # 108 + 1440 = 1548

n_z = nu * N                         # 90 decision variables
EPS = 1e-9                           # guards sqrt gradient at obstacle centre


def sfc_offset(j, m):
    """Start index in p of plane m at step j (4 values: n_x, n_y, n_z, b)."""
    return SFC_START + (j * M_MAX + m) * 4


def uav_dynamics_euler(x, u):
    """Forward Euler step of the simplified UAV model (identical to C++)."""
    px, py, pz = x[0], x[1], x[2]
    vx, vy, vz = x[3], x[4], x[5]
    phi, theta = x[6], x[7]

    T_thrust    = u[0]
    phi_ref_u   = u[1]
    theta_ref_u = u[2]

    ax =  T_thrust * cs.cos(phi) * cs.sin(theta)
    ay = -T_thrust * cs.sin(phi)
    az =  T_thrust * cs.cos(phi) * cs.cos(theta) - g

    return cs.vertcat(
        px + Ts * vx,
        py + Ts * vy,
        pz + Ts * vz,
        vx + Ts * (ax - Ax * vx),
        vy + Ts * (ay - Ay * vy),
        vz + Ts * (az - Az * vz),
        phi   + Ts / tau_phi   * (K_phi   * phi_ref_u   - phi),
        theta + Ts / tau_theta * (K_theta * theta_ref_u - theta),
    )


def build_problem():
    z = cs.SX.sym('z', n_z)
    p = cs.SX.sym('p', N_PARAMS)

    x0     = p[0:8]
    u_prev = p[8:11]

    p_ref_start = 11
    p_obs = p[101:104]
    v_obs = p[104:107]
    r_obs = p[107]

    u_ref = cs.vertcat(*u_ref_vals)

    cost        = 0.0
    g_hard_list = []   # ALM hard inequality constraints g_i(z,p) <= 0  (SFC)
    x_k         = x0

    for j in range(N):
        uj      = z[j * nu:(j + 1) * nu]
        uj_prev = u_prev if j == 0 else z[(j - 1) * nu:j * nu]

        # ---- State tracking ----
        p_ref_j = p[p_ref_start + j * 3:p_ref_start + j * 3 + 3]
        x_ref_j = cs.vertcat(p_ref_j, cs.SX.zeros(5, 1))
        dx = x_ref_j - x_k
        for i in range(nx):
            cost += Qx[i] * dx[i] ** 2

        # ---- Input effort ----
        du_ref = u_ref - uj
        for i in range(nu):
            cost += Qu[i] * du_ref[i] ** 2

        # ---- Input rate (smoothness) ----
        du = uj - uj_prev
        for i in range(nu):
            cost += QdU[i] * du[i] ** 2

        # ---- SOFT dynamic obstacle ----
        p_obs_j = p_obs + j * Ts * v_obs
        r_s_j   = R_S_MAX * j / N
        dp      = x_k[0:3] - p_obs_j
        dist    = cs.sqrt(cs.dot(dp, dp) + EPS)
        slack   = cs.fmax(0.0, (r_obs + r_s_j) - dist)
        cost += W_obs * slack ** 2

        # ---- HARD SFC half-space constraints (ALM) ----
        # n_jm . pos_k - b_jm <= 0   (inactive plane n=0,b=0 -> 0 <= 0)
        pos_k = x_k[0:3]
        for m in range(M_MAX):
            off  = sfc_offset(j, m)
            n_jm = p[off:off + 3]
            b_jm = p[off + 3]
            g_hard_list.append(cs.dot(n_jm, pos_k) - b_jm)

        # ---- Roll-out ----
        x_k = uav_dynamics_euler(x_k, uj)

    # ---- Terminal tracking ----
    p_ref_N = p[p_ref_start + (N - 1) * 3:p_ref_start + (N - 1) * 3 + 3]
    x_ref_N = cs.vertcat(p_ref_N, cs.SX.zeros(5, 1))
    dx = x_ref_N - x_k
    for i in range(nx):
        cost += Qx[i] * dx[i] ** 2

    # ---- Terminal SOFT obstacle ----
    p_obs_N = p_obs + N * Ts * v_obs
    dp      = x_k[0:3] - p_obs_N
    dist_N  = cs.sqrt(cs.dot(dp, dp) + EPS)
    slack_N = cs.fmax(0.0, (r_obs + R_S_MAX) - dist_N)
    cost += W_obs * slack_N ** 2

    # ---- Terminal HARD SFC (reuse step N-1 planes) ----
    pos_N = x_k[0:3]
    for m in range(M_MAX):
        off  = sfc_offset(N - 1, m)
        n_jm = p[off:off + 3]
        b_jm = p[off + 3]
        g_hard_list.append(cs.dot(n_jm, pos_N) - b_jm)

    F1 = cs.vertcat(*g_hard_list)
    return z, p, cost, F1


def generate():
    z, p, cost, F1 = build_problem()
    n_F1 = F1.shape[0]
    print(f"[INFO] # hard ALM (SFC) constraints F1: {n_F1}")

    # Only box hard constraint on the decision variable.
    bounds = og.constraints.Rectangle(u_min * N, u_max * N)

    # F1 <= 0 via set_c = Rectangle([-inf, 0])
    inf = float("inf")
    set_c = og.constraints.Rectangle([-inf] * n_F1, [0.0] * n_F1)
    set_y = og.constraints.BallInf(None, 1.0e6)

    problem = (
        og.builder.Problem(z, p, cost)
        .with_aug_lagrangian_constraints(F1, set_c, set_y)
        .with_constraints(bounds)
    )

    meta = og.config.OptimizerMeta().with_optimizer_name("nmpc_uav_obstacle")

    build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "nmpc_solver")

    build_config = (
        og.config.BuildConfiguration()
        .with_build_directory(build_dir)
        .with_build_mode("release")
        .with_build_c_bindings()
    )

    # ALM tuning (mirrors the original hard-constraint version).
    solver_config = (
        og.config.SolverConfiguration()
        .with_tolerance(1e-4)
        .with_delta_tolerance(1e-3)            # SFC feasibility tolerance
        .with_max_outer_iterations(10)
        .with_max_inner_iterations(800)
        .with_penalty_weight_update_factor(5.0)
        .with_initial_penalty(20.0)
        .with_sufficient_decrease_coefficient(0.7)
    )

    builder = og.builder.OpEnOptimizerBuilder(
        problem, meta, build_config, solver_config
    )
    builder.build()

    print("\n[OK] Solver compiled: SOFT obstacle + HARD (ALM) SFC, no PANOC-only.")
    print(f"     N_PARAMS         : {N_PARAMS}")
    print(f"     decision vars    : {n_z}")
    print(f"     M_MAX (per step) : {M_MAX}")
    print(f"     SFC block size   : {SFC_BLOCK}  (4 * {M_MAX} * {N})")
    print(f"     SFC start index  : {SFC_START}")
    print(f"     # hard (ALM) F1  : {n_F1}")
    print(f"     W_obs (soft obs) : {W_obs}")
    print()


if __name__ == "__main__":
    generate()