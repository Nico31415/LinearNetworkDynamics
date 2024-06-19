import seaborn as sns
import matplotlib as mpl
import numpy as np


class BlindColours:
    def __init__(self, reverse_cmap=True):
        sns.set_style("ticks", {
            'xtick.bottom': True,
            'xtick.top': False,
            'ytick.left': True,
            'ytick.right': False,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'xtick.color': '.1',
            'ytick.color': '.1',
        })

        sns.set_context("talk")

        hex_colours = ["#d65c00", "#0071b2", "#009e73", "#cc78a6", "#e59c00", "#55b2e8", "#efe440", "#000000"]
        self.blind_colours = [mpl.colors.to_rgb(h) for h in hex_colours]

        div = ['#6d0000', '#720400', '#770900', '#7c0d00', '#821200', '#871600', '#8b1b00', '#901f00', '#952300',
               '#9a2700', '#9f2c00', '#a33000', '#a83400', '#ad3800', '#b13c00', '#b64000', '#bb4500', '#bf4900',
               '#c44d00', '#c85100', '#cc5604', '#cf5b09', '#d3600e', '#d66513', '#d96a18', '#dd6f1d', '#e07422',
               '#e37927', '#e67e2c', '#ea8331', '#ed8836', '#f08d3b', '#f3923f', '#f69744', '#f99b49', '#fda04e',
               '#ffa555', '#feac62', '#fdb26e', '#fdb87a', '#fcbe87', '#fbc492', '#faca9e', '#facfa9', '#f9d5b4',
               '#f8dabf', '#f8e0ca', '#f7e5d5', '#f6ebe0', '#f6f0ea', '#ecf2f6', '#e3eef7', '#d9ebf8', '#d0e7f8',
               '#c6e4f9', '#bde0fa', '#b3ddfb', '#a9d9fc', '#9fd6fd', '#95d2fe', '#8bceff', '#85cafc', '#80c6f9',
               '#7bc2f6', '#75bef2', '#70baef', '#6bb6ec', '#66b1e9', '#61ade5', '#5ba9e2', '#56a5df', '#51a1dc',
               '#4c9dd8', '#4799d5', '#4295d2', '#3c91cf', '#378dcb', '#3289c8', '#2d85c5', '#2881c2', '#237dbf',
               '#2079ba', '#1e75b6', '#1c71b1', '#1a6dad', '#1969a8', '#1765a4', '#15619f', '#135d9b', '#115996',
               '#105592', '#0e518e', '#0c4d89', '#0a4a85', '#094681', '#07427d', '#053e79', '#033b74', '#023770',
               '#00346c']
        if reverse_cmap:
            div.reverse()
        self.div_cmap = mpl.colors.ListedColormap(div)

        oranges = [mpl.colors.to_rgb(h) for h in ['#871500', '#a93700', '#cc5400', '#ef721c', '#ff9c4a']]
        blues = [mpl.colors.to_rgb(h) for h in ['#00356e', '#005492', '#0975b7', '#4895d9', '#70b6fd']]
        greens = [mpl.colors.to_rgb(h) for h in ['#003e1d', '#005e39', '#008057', '#09a378', '#46c698']]
        self.colour_steps = [oranges, blues, greens]

    def get_colours(self):
        return self.blind_colours

    def get_div_cmap(self):
        return self.div_cmap

    def get_colour_steps(self):
        return self.colour_steps


def zero_balanced_weights(in_dim, hidden_dim, out_dim, sigma):
    r, _, _ = np.linalg.svd(np.random.normal(0., 1., (hidden_dim, hidden_dim)))

    w1 = np.random.normal(0., sigma, (hidden_dim, in_dim))
    w2 = np.random.normal(0., sigma, (out_dim, hidden_dim))
    u, s, vt = np.linalg.svd(w2 @ w1, False)

    s = np.diag(np.sqrt(s) * 1.15)

    smaller_dim = in_dim if in_dim < out_dim else out_dim

    s0 = np.vstack([s, np.zeros((hidden_dim - smaller_dim, smaller_dim))])
    w1 = r @ s0 @ vt

    s0 = np.hstack([s, np.zeros((smaller_dim, hidden_dim - smaller_dim))])
    w2 = u @ s0 @ r.T

    return w1, w2


def balanced_weights(hidden_dim, sigma=1, lmda=1):
    # generate random orthogonal matrix r of size (hidden_dim, hidden_dim)
    U, S, V = np.linalg.svd(np.random.randn(hidden_dim, hidden_dim))
    r = U @ V.T

    w1 = sigma * np.random.randn(hidden_dim, hidden_dim)
    w2 = sigma * np.random.randn(hidden_dim, hidden_dim)

    U_, S_, V_ = np.linalg.svd(w2 @ w1)
    s = np.sqrt(np.diag(S_))

    lmda = np.trace(w2 @ w1) / hidden_dim

    factor = (- lmda + np.sqrt(lmda ** 2 + 4 * s ** 2)) / 2

    s_2 = np.sqrt(np.diag(np.diag(factor)))
    s_1 = np.diag(np.diag(s) / np.diag(s_2))

    w1_out = r @ s_1 @ V.T
    w2_out = U @ s_2 @ r.T
    S_test = s_2 @ s_1

    q = w1_out @ w1_out.T - w2_out.T @ w2_out

    scale_by = lmda / q[0][0]
    w1_out = scale_by * w1_out
    w2_out = scale_by * w2_out
    q = w1_out @ w1_out.T - w2_out.T @ w2_out

    return w1_out, w2_out, S_test, q


print(balanced_weights(3))