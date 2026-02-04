#%%
import torch
import plotly.graph_objects as go

# Reproducibility
torch.manual_seed(0)

# Parameters
n = 30
dim = 3
sigma = 1.0

xs = torch.linspace(-2, 2, 30)
ys = torch.linspace(-2, 2, 30)

def gaussian_kernel(x, y, sigma=1.0):
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    return torch.exp(-((x - y) ** 2).sum(2) / (2 * sigma ** 2))

def compute_mmd(x, y):
    return (
        gaussian_kernel(x, x, sigma).mean()
        + gaussian_kernel(y, y, sigma).mean()
        - 2 * gaussian_kernel(x, y, sigma).mean()
    )

# Fixed source distribution X ~ N(0, I)
X = torch.randn(n, dim)

# Compute MMD surface
X_shift, Y_shift = torch.meshgrid(xs, ys, indexing="ij")
MMD = torch.zeros_like(X_shift)

for i in range(len(xs)):
    for j in range(len(ys)):
        shift = torch.zeros(dim)
        shift[0] = xs[i]
        shift[1] = ys[j]
        Y = torch.randn(n, dim) + shift
        MMD[i, j] = compute_mmd(X, Y)

# Interactive 3D surface
fig = go.Figure(
    data=[
        go.Surface(
            x=X_shift.numpy(),
            y=Y_shift.numpy(),
            z=MMD.numpy(),
        )
    ]
)

fig.update_layout(
    title="Interactive 3D Surface of MMD(x, y)",
    scene=dict(
        xaxis_title="x shift",
        yaxis_title="y shift",
        zaxis_title="MMD(x, y)",
    ),
)

fig.show()

