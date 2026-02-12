import numpy as np
import tkinter as tk
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import math

# ============================================================================
#               FIXED-POINT Q16.16 HELPERS + CORDIC sin/cos
# ============================================================================

SHIFT = 16
SCALE = 1 << SHIFT

def to_q16(f: float) -> int:
    return int(round(f * SCALE))

def from_q16(q: int) -> float:
    return q / SCALE

def q16_mul(a: int, b: int) -> int:
    return (a * b) >> SHIFT

def q16_div(a: int, b: int) -> int:
    return (a << SHIFT) // b if b != 0 else 0

# Precomputed for CORDIC: N iterations
CORDIC_N = 18
CORDIC_ANGLES = [to_q16(math.atan(2**-i)) for i in range(CORDIC_N)]
CORDIC_K = to_q16(1.0 / math.prod(math.sqrt(1 + 2**(-2*i)) for i in range(CORDIC_N)))

def q16_cordic_cos_sin(theta: int) -> tuple[int, int]:
    """ CORDIC cos and sin in Q16, input theta in radians * SCALE, range -pi/2 to pi/2 """
    # Handle negative angles for symmetry
    sign = 1
    if theta < 0:
        theta = -theta
        sign = -1

    x = CORDIC_K
    y = 0
    z = theta

    for i in range(CORDIC_N):
        if z >= 0:
            x_new = x - (y >> i)  # arithmetic shift ok in Python for neg
            y_new = y + (x >> i)
            z -= CORDIC_ANGLES[i]
        else:
            x_new = x + (y >> i)
            y_new = y - (x >> i)
            z += CORDIC_ANGLES[i]
        x, y = x_new, y_new

    return x, y * sign  # cos is even, sin is odd

# Tunables
FX_ETA   = to_q16(0.22)
FX_GATE  = to_q16(0.18)  # threshold for angle / 90-ish
FX_GAMMA = to_q16(0.989)
NOISE_STD = 0.04  # noise level for observations

# ============================================================================
#                   QUATERNION OPERATIONS
# ============================================================================

def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [
        q16_mul(w1,w2) - q16_mul(x1,x2) - q16_mul(y1,y2) - q16_mul(z1,z2),
        q16_mul(w1,x2) + q16_mul(x1,w2) + q16_mul(y1,z2) - q16_mul(z1,y2),
        q16_mul(w1,y2) - q16_mul(x1,z2) + q16_mul(y1,w2) + q16_mul(z1,x2),
        q16_mul(w1,z2) + q16_mul(x1,y2) - q16_mul(y1,x2) + q16_mul(z1,w2)
    ]

def quat_conj(q):
    return [q[0], -q[1], -q[2], -q[3]]

def quat_dot(a, b):
    return sum(q16_mul(a[i], b[i]) for i in range(4))

def quat_norm(q):
    n2 = sum(q16_mul(x,x) for x in q)
    if n2 <= 0: return [SCALE, 0, 0, 0]
    n = int(math.sqrt(from_q16(n2)) * SCALE + 0.5) or SCALE
    inv = q16_div(SCALE, n)
    return [q16_mul(x, inv) for x in q]

def geodesic_angle_deg(q1, q2):
    dot = quat_dot(q1, q2)
    dot = max(min(dot, SCALE), -SCALE)
    cos_half = from_q16(dot)
    half_rad = math.acos(np.clip(cos_half, -1.0, 1.0))
    return math.degrees(2 * half_rad)

def exp_map(v_vec):  # v_vec = [x,y,z] theta * axis / 2 in Q16
    theta_sq = sum(q16_mul(vi, vi) for vi in v_vec)
    if theta_sq < to_q16(0.0005)**2:
        half = [vi >> 1 for vi in v_vec]
        return [SCALE] + half

    theta = int(math.sqrt(from_q16(theta_sq)) * SCALE + 0.5)
    if theta == 0: theta = 1

    half_theta = theta >> 1
    c = q16_cordic_cos_sin(half_theta)[0]
    s = q16_cordic_cos_sin(half_theta)[1]

    sin_over_theta = q16_div(s, theta)
    unit = [q16_div(vi, theta) for vi in v_vec]  # vi / theta = axis component

    vec_part = [q16_mul(sin_over_theta, u) for u in unit]
    return [c] + vec_part

def get_axis_from_delta(delta):
    # delta = [cos(theta/2), sin(theta/2) * axis_x, ...]
    # axis = delta[1:] / sin(theta/2) if sin !=0
    half_theta = to_q16(math.acos(np.clip(from_q16(delta[0]), -1.0, 1.0)))
    sin_half = q16_cordic_cos_sin(half_theta)[1]
    if abs(sin_half) < 10:  # very small
        return [0.0, 0.0, 1.0]  # arbitrary
    axis = [from_q16(q16_div(delta[i], sin_half)) for i in range(1,4)]
    return axis

# ============================================================================
#                         ADAPTIVE FILTER
# ============================================================================

class FrameworkProver:
    def __init__(self):
        self.q = [SCALE, 0, 0, 0]
        self.alpha = SCALE
        self.history_angle = []
        self.history_proxy = []
        self.history_alpha = []
        self.history_axis_x = []
        self.history_axis_y = []
        self.history_axis_z = []
        self.novel_flags = []

    def step(self, obs_q):
        angle_deg = geodesic_angle_deg(self.q, obs_q)
        proxy = SCALE - quat_dot(self.q, obs_q)
        is_novel = angle_deg > from_q16(FX_GATE) * 90

        q_inv = quat_conj(self.q)
        delta = quat_mul(q_inv, obs_q)

        # Log axis
        axis = get_axis_from_delta(delta)
        self.history_axis_x.append(axis[0])
        self.history_axis_y.append(axis[1])
        self.history_axis_z.append(axis[2])

        # Conservative step: scale by alpha / 4
        vec = [q16_mul(x, self.alpha >> 2) for x in delta[1:]]
        exp_delta = exp_map(vec)

        self.q = quat_mul(self.q, exp_delta)
        self.q = quat_norm(self.q)

        self.alpha = q16_mul(self.alpha, FX_GAMMA)

        self.history_angle.append(angle_deg)
        self.history_proxy.append(from_q16(proxy))
        self.history_alpha.append(from_q16(self.alpha))
        self.novel_flags.append(is_novel)

        return is_novel, angle_deg


# ============================================================================
#                           VISUALIZER
# ============================================================================

class ProofGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lie Group Adaptive Filter • Ultimate 2026")
        self.canvas = tk.Canvas(root, width=1100, height=580, bg="#0b0e14")
        self.canvas.pack(pady=12)

        self.prover = FrameworkProver()
        self.step_count = 0
        self.max_steps = 1000
        self.trajectory = []

        self.label = tk.Label(root, text="Running...", fg="#8f8", bg="#111", font=("Consolas", 13))
        self.label.pack()

        self.rng = random.Random(42)
        self.run()

    def get_observation(self):
        t = self.step_count * 0.024

        if self.step_count < 280:
            theta = t
            phi = 1.7 * t
            z_pert = 0.35 * math.sin(2 * theta) * math.cos(phi)
            w = math.sqrt(max(0, 1 - z_pert**2))
            obs_f = [w, math.cos(theta)*z_pert, math.sin(theta)*z_pert, 0]
        elif self.step_count < 380:
            obs_f = [0.6533, 0, 0.7568, 0]
        elif self.step_count < 620:
            if not hasattr(self, 'rw_q_f'):
                self.rw_q_f = [1.0, 0, 0, 0]
            dw = [self.rng.gauss(0, 0.028) for _ in range(3)]
            theta_dw = math.sqrt(sum(d**2 for d in dw))
            if theta_dw > 1e-6:
                c = math.cos(theta_dw / 2)
                s = math.sin(theta_dw / 2) / theta_dw
                exp_dw_f = [c] + [s * d for d in dw]
            else:
                exp_dw_f = [1.0, 0, 0, 0]
            self.rw_q_f = [  # simple float quat mul
                self.rw_q_f[0]*exp_dw_f[0] - self.rw_q_f[1]*exp_dw_f[1] - self.rw_q_f[2]*exp_dw_f[2] - self.rw_q_f[3]*exp_dw_f[3],
                self.rw_q_f[0]*exp_dw_f[1] + self.rw_q_f[1]*exp_dw_f[0] + self.rw_q_f[2]*exp_dw_f[3] - self.rw_q_f[3]*exp_dw_f[2],
                self.rw_q_f[0]*exp_dw_f[2] - self.rw_q_f[1]*exp_dw_f[3] + self.rw_q_f[2]*exp_dw_f[0] + self.rw_q_f[3]*exp_dw_f[1],
                self.rw_q_f[0]*exp_dw_f[3] + self.rw_q_f[1]*exp_dw_f[2] - self.rw_q_f[2]*exp_dw_f[1] + self.rw_q_f[3]*exp_dw_f[0]
            ]
            norm = math.sqrt(sum(x**2 for x in self.rw_q_f))
            obs_f = [x / norm for x in self.rw_q_f]
        else:
            obs_f = [math.cos(0.9*t), math.sin(0.9*t), 0.4*math.sin(2.3*t), 0]
            norm = math.sqrt(sum(x**2 for x in obs_f))
            obs_f = [x / norm for x in obs_f]

        # Add noise
        obs_f = [x + self.rng.gauss(0, NOISE_STD) for x in obs_f]
        norm = math.sqrt(sum(x**2 for x in obs_f)) or 1.0
        obs_f = [x / norm for x in obs_f]

        return [to_q16(x) for x in obs_f]

    def run(self):
        if self.step_count >= self.max_steps:
            self.label.config(text="Finished — summary shown & saved", fg="#fa0")
            self.plot_summary()
            return

        obs = self.get_observation()
        novel, angle = self.prover.step(obs)
        self.trajectory.append(np.array([from_q16(x) for x in self.prover.q]))

        self.draw(novel, angle)
        self.step_count += 1
        self.root.after(10, self.run)

    def draw(self, is_novel, angle_deg):
        self.canvas.delete("all")
        h = self.prover.history_angle
        if len(h) < 2: return

        scale_y = 5.0
        pts = [(i * (1100 / self.max_steps), 560 - v * scale_y) for i,v in enumerate(h)]

        color = "#ff5555" if is_novel else "#55ff99"
        self.canvas.create_line(pts, fill=color, width=2)

        thresh_deg = from_q16(FX_GATE) * 90
        ty = 560 - thresh_deg * scale_y
        self.canvas.create_line(0, ty, 1100, ty, fill="#667799", dash=(5,5))

        status = f"NOVEL  ({angle_deg:.1f}°)" if is_novel else f"track  ({angle_deg:.1f}°)"
        self.label.config(text=f"Step {self.step_count:4d} | {status} | α={self.prover.history_alpha[-1]:.4f}",
                          fg="#ff7733" if is_novel else "#44dd88")

    def plot_summary(self):
        if not self.trajectory: return
        traj = np.vstack(self.trajectory)

        fig = plt.figure(figsize=(15, 12), facecolor='#0e111a')
        fig.suptitle("Lie Group Adaptive Tracking – Ultimate Version", color='white', fontsize=16)

        ax1 = fig.add_subplot(3, 2, 1)
        ax1.plot(self.prover.history_angle, color='#aaccff', lw=1.3)
        ax1.axhline(from_q16(FX_GATE)*90, color='#ff6666', ls='--', alpha=0.8)
        ax1.set_facecolor('#121622')
        ax1.tick_params(colors='w')
        ax1.spines[:].set_color('gray')
        ax1.set_title("Angular Distance 2∠ (deg)", color='w')
        ax1.set_ylabel("degrees", color='w')

        ax2 = fig.add_subplot(3, 2, 2)
        ax2.plot(self.prover.history_alpha, color='#66ffaa', lw=1.5)
        ax2.set_facecolor('#121622')
        ax2.tick_params(colors='w')
        ax2.spines[:].set_color('gray')
        ax2.set_title("Adaptive Gain α", color='w')

        ax3 = fig.add_subplot(3, 2, 3, projection='3d')
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(traj)
        ax3.plot(reduced[:,0], reduced[:,1], reduced[:,2], color='#cc88ff', lw=1.8, alpha=0.92)
        ax3.set_facecolor('#0a0c14')
        ax3.xaxis.line.set_color('gray')
        ax3.yaxis.line.set_color('gray')
        ax3.zaxis.line.set_color('gray')
        ax3.tick_params(colors='w')
        ax3.set_title("Trajectory on S³ (PCA)", color='w')

        ax4 = fig.add_subplot(3, 2, 4)
        counts = [self.prover.novel_flags.count(False), self.prover.novel_flags.count(True)]
        ax4.bar(['Stable', 'Novel'], counts, color=['#55ff99','#ff5555'], alpha=0.9)
        ax4.set_facecolor('#121622')
        ax4.tick_params(colors='w')
        ax4.spines[:].set_color('gray')
        ax4.set_title("Novelty Detections", color='w')

        ax5 = fig.add_subplot(3, 2, (5,6))
        ax5.plot(self.prover.history_axis_x, label='axis x', color='#ff8888')
        ax5.plot(self.prover.history_axis_y, label='axis y', color='#88ff88')
        ax5.plot(self.prover.history_axis_z, label='axis z', color='#8888ff')
        ax5.set_facecolor('#121622')
        ax5.tick_params(colors='w')
        ax5.spines[:].set_color('gray')
        ax5.legend(loc='upper right', labelcolor='w')
        ax5.set_title("Rotation Axis Over Time", color='w')
        ax5.set_ylabel("component", color='w')
        ax5.set_ylim(-1.1, 1.1)

        fig.tight_layout(rect=[0,0,1,0.94])
        plt.savefig('lie_group_summary.png')
        plt.show()

        # Print summary
        print(f"Mean angular distance: {np.mean(self.prover.history_angle):.2f} deg")
        print(f"Max angular distance: {np.max(self.prover.history_angle):.2f} deg")
        print(f"Novelty detections: {sum(self.prover.novel_flags)} / {self.max_steps}")
        print(f"Final alpha: {self.prover.history_alpha[-1]:.4f}")
        print("Summary plot saved to 'lie_group_summary.png'")


if __name__ == "__main__":
    root = tk.Tk()
    root.configure(bg="#111")
    app = ProofGUI(root)
    root.mainloop()