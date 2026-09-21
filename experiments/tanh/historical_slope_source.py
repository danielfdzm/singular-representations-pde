import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

# -----------------------------
# Plot styling
# -----------------------------
mpl.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "axes.linewidth": 1.2,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "lines.linewidth": 2.0,
    "legend.fontsize": 13,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.fancybox": False,
    "figure.constrained_layout.use": False,
})

BLUE = "#0000FF"
RED = "#FF0000"
BLACK = "#111111"
GREEN = "green"

def style_legend(ax, loc="best", fontsize=14):
    leg = ax.legend(
        loc=loc,
        fontsize=fontsize,
        handlelength=1.2,
        handletextpad=0.4,
        labelspacing=0.2,
        borderpad=0.25,
    )
    if leg is not None:
        frame = leg.get_frame()
        frame.set_edgecolor("black")
        frame.set_linewidth(1.4)
    return leg

# -----------------------------
# Problem setup
# -----------------------------
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

R = 10.0
eps = 1.0
N_int = 2001
N_plot = 4001

USE_HALF = True

def u_true(x):
    s = 1.0 / torch.cosh(x) ** 2
    return x * s

def f_true(x):
    s = 1.0 / torch.cosh(x) ** 2
    t = torch.tanh(x)
    return 4.0 * s * t - 3.0 * x * s + 6.0 * x * s * s

def nn_u(x, w1, a1, b1, w2, a2, b2):
    return w1 * torch.tanh(a1 * x + b1) + w2 * torch.tanh(a2 * x + b2)

def trapezoid_weights(R, N, device=None, dtype=None):
    dx = 2.0 * R / (N - 1)
    w = torch.ones(N, device=device, dtype=dtype) * dx
    w[0] *= 0.5
    w[-1] *= 0.5
    return w

def energy_functional(R, N, w1, a1, b1, w2, a2, b2, use_half=True):
    x = torch.linspace(-R, R, N, requires_grad=True)

    u = nn_u(x, w1, a1, b1, w2, a2, b2)
    du_dx = torch.autograd.grad(u.sum(), x, create_graph=True)[0]

    f = f_true(x.detach())

    if use_half:
        integrand = 0.5 * (du_dx**2 + u**2) - f * u
    else:
        integrand = (du_dx**2 + u**2) - f * u

    wts = trapezoid_weights(R, N, device=x.device, dtype=x.dtype)
    one_d = torch.sum(integrand * wts)
    return (2.0 * R) * one_d

# -----------------------------
# Initialize parameters
# -----------------------------
w1 = torch.nn.Parameter(torch.tensor(1.0 / (2.0 * eps) + 3))
w2 = torch.nn.Parameter(torch.tensor(-1.0 / (2.0 * eps) + 2))
a1 = torch.nn.Parameter(torch.tensor(2.0))
a2 = torch.nn.Parameter(torch.tensor(0.5))
b1 = torch.nn.Parameter(torch.tensor(0.0))
b2 = torch.nn.Parameter(torch.tensor(0.0))

params = [w1, a1, b1, w2, a2, b2]

# -----------------------------
# Helpers
# -----------------------------
def get_param_state():
    return [p.detach().clone() for p in params]

def set_param_state(state):
    with torch.no_grad():
        for p, value in zip(params, state):
            p.copy_(value)

def save_snapshot(label):
    print(
        label,
        f"w1={w1.item(): .6f}",
        f"a1={a1.item(): .6f}",
        f"b1={b1.item(): .6f}",
        f"w2={w2.item(): .6f}",
        f"a2={a2.item(): .6f}",
        f"b2={b2.item(): .6f}",
    )

    snapshots.append({
        "label": label,
        "w1": w1.detach().clone(),
        "a1": a1.detach().clone(),
        "b1": b1.detach().clone(),
        "w2": w2.detach().clone(),
        "a2": a2.detach().clone(),
        "b2": b2.detach().clone(),
    })

def run_lbfgs(max_iter):
    opt = torch.optim.LBFGS(
        params,
        lr=1.0,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        L = energy_functional(R, N_int, w1, a1, b1, w2, a2, b2, use_half=USE_HALF)
        L.backward()
        return L

    opt.step(closure)

# -----------------------------
# Optimize and save snapshots
# Columns: initial, 1/5 Adam, final Adam, 1/5 LBFGS, final LBFGS
# -----------------------------
n_adam = 5000
lbfgs_max_iter = 1000
lbfgs_snapshot_iter = lbfgs_max_iter // 5

snapshots = []

save_snapshot("Initial")

# Adam training
opt_adam = torch.optim.Adam(params, lr=1e-3)

for it in range(1, n_adam + 1):
    opt_adam.zero_grad()
    loss = energy_functional(
        R,
        N_int // 2 + 1,
        w1,
        a1,
        b1,
        w2,
        a2,
        b2,
        use_half=USE_HALF,
    )
    loss.backward()
    opt_adam.step()

    if it == n_adam // 3:
        save_snapshot("1/5 Adam")

save_snapshot("Final Adam")

adam_final_state = get_param_state()

# 1/5 LBFGS snapshot from Adam-final state
set_param_state(adam_final_state)
run_lbfgs(lbfgs_snapshot_iter)
save_snapshot("1/5 LBFGS")

# Final LBFGS from the same Adam-final state.
# This makes the final column independent of lbfgs_snapshot_iter.
set_param_state(adam_final_state)
run_lbfgs(lbfgs_max_iter)
save_snapshot("Final LBFGS")

# -----------------------------
# Evaluate ground truth
# -----------------------------
x_plot = torch.linspace(-R, R, N_plot)
x_np = x_plot.cpu().numpy()

with torch.no_grad():
    u_gt = u_true(x_plot).cpu().numpy()

# -----------------------------
# Plotting: remove the initial left column
# Row 1: individual neurons
# Row 2: NN solution vs ground truth
# -----------------------------
plot_snapshots = snapshots[1:]
legend_col = 0
fig, axes = plt.subplots(2, len(plot_snapshots), figsize=(18, 7))

for col, snap in enumerate(plot_snapshots):
    w1_s = snap["w1"]
    a1_s = snap["a1"]
    b1_s = snap["b1"]
    w2_s = snap["w2"]
    a2_s = snap["a2"]
    b2_s = snap["b2"]

    with torch.no_grad():
        neuron1 = (w1_s * torch.tanh(a1_s * x_plot + b1_s)).cpu().numpy()
        neuron2 = (w2_s * torch.tanh(a2_s * x_plot + b2_s)).cpu().numpy()
        u_nn = neuron1 + neuron2

    ax = axes[0, col]
    ax.plot(x_np, neuron1, color=BLUE, lw=2.0, label=r"$w_1\,\tanh(a_1 x + b_1)$")
    ax.plot(x_np, neuron2, color=RED, lw=2.0, label=r"$w_2\,\tanh(a_2 x + b_2)$")
    ax.axhline(0.0, color="black", lw=0.8, ls="--")

    ax.set_xlim(-R, R)
    ax.set_ylim(-30, 30)
    ax.set_xticks([])
    ax.tick_params(direction="out", length=4.5, width=1.1)

    if col == 0:
        ax.set_ylabel(r"$u_i(x)$")
        ax.set_yticks([-20, -10, 0, 10, 20])
    else:
        ax.set_yticks([])
    if col == legend_col:
        style_legend(ax, loc="upper right", fontsize=14)

    ax = axes[1, col]
    ax.plot(
        x_np,
        u_nn,
        color=BLACK,
        lw=2.0,
        ls="--",
        label=r"$w_1\tanh(a_1x+b_1)$" + "\n" + r"$+w_2\tanh(a_2x+b_2)$",
    )
    ax.plot(x_np, u_gt, color=GREEN, lw=2.0, label=r"$x\,\tanh'(x)$")
    ax.axhline(0.0, color="black", lw=0.8, ls="--")

    ax.set_xlabel(r"$x$")
    ax.set_xlim(-R, R)
    ax.set_ylim(-1, 1)
    ax.set_xticks([-5, 0, 5])
    ax.tick_params(direction="out", length=4.5, width=1.1)

    if col == 0:
        ax.set_ylabel(r"$u(x)$")
        ax.set_yticks([-0.5, 0, 0.5])
    else:
        ax.set_yticks([])
    if col == legend_col:
        style_legend(ax, loc="upper right", fontsize=14)

for row in range(2):
    for col in range(len(plot_snapshots) - 1):
        ax_right = axes[row, col + 1]
        ax_right.plot(
            -0.04,
            0.5,
            marker=">",
            color="gray",
            markersize=16,
            transform=ax_right.transAxes,
            clip_on=False,
            linestyle="None",
        )

plt.tight_layout()
fig.subplots_adjust(wspace=0.08)

out_pdf = "tanh_x_tanhprime_energy_minimization.pdf"
fig.savefig(out_pdf, format="pdf", dpi=1200, bbox_inches="tight")
plt.show()

print("Final parameters:")
print(f"  w1 = {w1.item(): .6f}, a1 = {a1.item(): .6f}, b1 = {b1.item(): .6f}")
print(f"  w2 = {w2.item(): .6f}, a2 = {a2.item(): .6f}, b2 = {b2.item(): .6f}")
print(f"Saved figure to: {out_pdf}")