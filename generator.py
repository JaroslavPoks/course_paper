import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.interpolate import CubicSpline
import random


def spline_check(x, y, l_b, u_b):
    l_bc = 0
    r_bc = 0

    cs = CubicSpline(x, y, bc_type=((1, [l_bc]), (1, [r_bc])))
    if max(cs(points)) < u_b and min(cs(points)) > l_b:
        return 1
    else:
        return 0


def coef_generator(n, segment):
    line = np.array([])
    samples_t = np.array([])
    for i in range(4):
        for j in range(n):
            elem = random.uniform(segment[j], segment[j + 1])
            line = np.append(line, elem)

        np.random.shuffle(line)
        samples_t = np.append(samples_t, line)
        line = np.array([])

    samples_t = np.reshape(samples_t, (n, 4, 1))
    return samples_t


def spline_graph(x, y):
    l_bc = y[4]
    r_bc = y[5]
    cs = CubicSpline(x, y[0:4], bc_type=((1, l_bc), (1, r_bc)))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    plt.scatter(x, y[0:4])
    plt.plot(points, cs(points))
    ax.set_ylim(0.0, 0.5)
    plt.show()


# f = open('samples_poisson.npy', 'w')
n = 8192
m = 2048

l_bound_y = 0.2
r_bound_y = 5.0
l_bound_p = 0.02
r_bound_p = 0.48

# poisson = np.zeros((n + m, 4, 1))
# poisson[:, :, 0] = 0.3
boundary = np.zeros((n + m, 2, 2))
#temp[:, 1, 0] = 1
#line = np.array([])
nods = np.array([0., 0.333, 0.666, 1.])
points = np.linspace(0, 1, 200)

segment1_y = np.linspace(l_bound_y, r_bound_y, int(n*1.15) + 1)
segment2_y = np.linspace(l_bound_y, r_bound_y, int(m*1.15) + 1)

segment1_p = np.linspace(l_bound_p, r_bound_p, int(n*1.15) + 1)
segment2_p = np.linspace(l_bound_p, r_bound_p, int(m*1.15) + 1)

a1 = 0
samples1_y = coef_generator(int(n*1.1), segment1_y)
mask = np.array([])
for elem in samples1_y:
    a1 += spline_check(nods, elem, 0.2, 5.0)
    mask = np.append(mask, spline_check(nods, elem, 0.2, 5.0))
mask = np.array(mask, dtype='bool')
samples1_y = samples1_y[mask]
samples1_y = samples1_y[0:n]
print(a1)

a2 = 0
samples2_y = coef_generator(int(m*1.1), segment2_y)
mask = np.array([])
for elem in samples2_y:
    a2 += spline_check(nods, elem, 0.2, 5.0)
    mask = np.append(mask, bool(spline_check(nods, elem, 0.2, 5.0)))
mask = np.array(mask, dtype='bool')
samples2_y = samples2_y[mask]
samples2_y = samples2_y[0:m]
print(a2)


a3 = 0
samples1_p = coef_generator(int(n*1.15), segment1_p)
mask = np.array([])
for elem in samples1_p:
    a3 += spline_check(nods, elem, 0.02, 0.48)
    mask = np.append(mask, spline_check(nods, elem, 0.02, 0.48))
mask = np.array(mask, dtype='bool')
samples1_p = samples1_p[mask]
samples1_p = samples1_p[0:n]
print(a3)

a4 = 0
samples2_p = coef_generator(int(m*1.15), segment2_p)
mask = np.array([])
for elem in samples2_p:
    a4 += spline_check(nods, elem, 0.02, 0.48)
    mask = np.append(mask, bool(spline_check(nods, elem, 0.02, 0.48)))
mask = np.array(mask, dtype='bool')
samples2_p = samples2_p[mask]
samples2_p = samples2_p[0:m]
print(a4)


samples = np.append(samples1_y, samples2_y, axis=0)
poisson = np.append(samples1_p, samples2_p, axis=0)
print(samples.shape)

# for i in range(len(samples)):
#     boundary[i, 0, 0] = (samples[i, 1, 0] - samples[i, 0, 0]) / 0.333
#     boundary[i, 1, 0] = (samples[i, 3, 0] - samples[i, 2, 0]) / 0.333

samples = np.append(samples, poisson, axis=2)
samples = np.append(samples, boundary, axis=1)

# for i in range(50):
#     spline_graph(nods, samples[i, :, 1])


# np.save('samples_poisson.npy', samples)
# f.close()
