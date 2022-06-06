import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math
import scipy.ndimage as ndimage


MAP_SIZE = 128
DROPLET_COUNT = 50
INIT_WATER_VOLUME = 2
SOLUBILITY = 0.5
WATER_EVAP_RATE = 0.85
INERTIA_LAND = 0.02
INERTIA_UNDERWATER = 0.5


def gen_perlin_noise(scales=[4, 8, 16, 32], factors=[1, 0.5, 0.25, 0.125], size=64):
    map = np.zeros((size, size))
    for scale, factor in zip(scales, factors):
        noise = np.random.rand(scale, scale) * factor
        noise = ndimage.zoom(noise, size/scale)
        map += noise
    map /= sum(factors)
    return map

def gen_basin(size=64):
    sigma = 0.6
    lin = np.linspace(-1, 1, size)
    lin = np.exp(-lin**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
    lin_min = lin[0]**2
    lin_max = lin[size//2]**2
    basin = np.matmul(lin[:,None], lin[None,:])
    basin = (basin - lin_min) / (lin_max - lin_min)
    return 1-basin

def get_interp_value(map, x, y):
    return ndimage.map_coordinates(map, [[y], [x]], order=2, mode='nearest')[0]

def get_interp_values(map, xs, ys):
    return ndimage.map_coordinates(map, [[ys], [xs]], order=2, mode='nearest').flatten()

def add_to_value(map: np.ndarray, xs: np.ndarray, ys: np.ndarray, values: np.ndarray):
    xs_lower = np.floor(xs)
    xs_upper = np.ceil(xs)
    ys_lower = np.floor(ys)
    ys_upper = np.ceil(ys)
    dist_xs_lower = np.abs(xs - xs_lower)
    dist_xs_upper = np.abs(xs - xs_upper)
    dist_ys_lower = np.abs(ys - ys_lower)
    dist_ys_upper = np.abs(ys - ys_upper)
    # dist_ll = np.sqrt(dist_xs_lower ** 2 + dist_ys_lower ** 2)
    # dist_lu = np.sqrt(dist_xs_upper ** 2 + dist_ys_lower ** 2)
    # dist_ul = np.sqrt(dist_xs_lower ** 2 + dist_ys_upper ** 2)
    # dist_uu = np.sqrt(dist_xs_upper ** 2 + dist_ys_upper ** 2)
    dist_ll = dist_xs_upper * dist_ys_upper
    dist_lu = dist_xs_lower * dist_ys_upper
    dist_ul = dist_xs_upper * dist_ys_lower
    dist_uu = dist_xs_lower * dist_ys_lower
    dist_sum = dist_ll + dist_lu + dist_ul + dist_uu
    dist_sum += 1e-6
    map[ys_lower.astype(int), xs_lower.astype(int)] += values * dist_ll / dist_sum
    map[ys_lower.astype(int), xs_upper.astype(int)] += values * dist_lu / dist_sum
    map[ys_upper.astype(int), xs_lower.astype(int)] += values * dist_ul / dist_sum
    map[ys_upper.astype(int), xs_upper.astype(int)] += values * dist_uu / dist_sum

def get_grads(map, xs, ys):
    x_grads = get_interp_values(map, xs+0.5, ys) - get_interp_values(map, xs-0.5, ys)
    y_grads = get_interp_values(map, xs, ys+0.5) - get_interp_values(map, xs, ys-0.5)
    return x_grads.flatten(), y_grads.flatten()

def update(map, d_ps, d_vs, d_ws, d_ss, trajectory_map):
    x_grads, y_grads = get_grads(map, d_ps[1], d_ps[0])
    grads = np.sqrt(x_grads ** 2 + y_grads ** 2)
    d_vs *= INERTIA_LAND
    d_vs[1] -= x_grads
    d_vs[0] -= y_grads
    d_vs /= (np.sqrt(d_vs[0]**2 + d_vs[1]**2) + 1e-6)
    new_ps = d_ps + d_vs
    resets = (new_ps < 0) + (new_ps > MAP_SIZE - 1)
    resets = resets[0] + resets[1]
    height_diffs = get_interp_values(map, d_ps[1], d_ps[0]) - get_interp_values(map, new_ps[1], new_ps[0])
    # sol
    gap = d_ws * SOLUBILITY - d_ss
    sols = gap ** 2
    sols = np.clip(np.min([sols, height_diffs], axis=0), 0, 100)

    # sed
    seds = -gap
    seds = np.clip(np.max([seds, height_diffs], axis=0), -100, 0)
    # print('@', sols, seds, resets)
    # print(sols[0], seds[0], sols[0] + seds[0])
    sols += seds
    # print(gap, sols, height_diffs, resets)
    sols *= 1-resets
    add_to_value(map, d_ps[1], d_ps[0], -sols)
    d_ws += sols * grads
    # print(sols)

    # reset some of the droplets
    resets = resets + (d_ws < 1e-3)
    # print(resets)

    add_to_value(trajectory_map, d_ps[1], d_ps[0], sols)

    d_ps += d_vs
    if sum(resets) > 0:
        d_ps[:,resets] = np.random.rand(2,sum(resets)) * MAP_SIZE - 1
        d_vs[:,resets] = 0
        d_ws[resets] = INIT_WATER_VOLUME
        d_ss[resets] = 0

    not_resets = 1 - resets
    if sum(not_resets) > 0:
        d_ws[not_resets] *= WATER_EVAP_RATE

map = 0.9 * gen_perlin_noise(size=MAP_SIZE) + 0.1 * gen_basin(size=MAP_SIZE)
original_map = map.copy()
trajectory_map = np.zeros_like(map)

# droplets' positions
d_ps = np.random.rand(2, DROPLET_COUNT) * MAP_SIZE - 1
# droplets' velocities
d_vs = np.zeros((2, DROPLET_COUNT))
# droplets' water count
d_ws = np.zeros(DROPLET_COUNT) + INIT_WATER_VOLUME
# droplets' sediment
d_ss = np.zeros(DROPLET_COUNT)

for i in range(10):
    update(map, d_ps, d_vs, d_ws, d_ss, trajectory_map)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# ag_scatter = axs.scatter(ys, xs, c='w', marker='.')
im1 = axs[0].imshow(map, cmap='terrain')
im2 = axs[1].imshow(trajectory_map)
ims = [im1, im2]

def animate(frame):
    global map, trajectory_map
    if frame == 0:
        map += 1e-3 * original_map - 5e-4
        # mx = np.max(map)
        # mi = np.min(map)
        # print(mx, mi)
        # map -= mi
        # map /= (mx - mi)
        mx = np.max(map)
        mi = np.min(map)
        print(mx, mi)

    for i in range(10):
        update(map, d_ps, d_vs, d_ws, d_ss, trajectory_map)
    ims[0].set_data(map)
    ims[1].set_data(trajectory_map)
    trajectory_map *= 0.99
    return ims

anim = animation.FuncAnimation(fig, animate, blit=True, frames=1000,
                               interval=5)
                               
plt.show()