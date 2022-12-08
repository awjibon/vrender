import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from scipy.interpolate import interpn
from scipy.spatial.transform import Rotation as R
from scipy import io
from scipy import ndimage as nimg

c_alpha_mask = 0.5
c_post_median = 3


def myTransferFunction(x, background=0.0):
    r = np.ones(shape=x.shape, dtype=np.float32)*background
    g = np.ones(shape=x.shape, dtype=np.float32)*background
    b = np.ones(shape=x.shape, dtype=np.float32)*background
    a = np.ones(shape=x.shape, dtype=np.float32)*0.000

    # mask
    #a[:] = x[:];
    r[:] = x[:]; g[:] = x[:]; b[:] = x[:]
    val = 1; a[x == val] = c_alpha_mask; r[x == val] = 1.0; g[x == val] = 1.0; b[x == val] = 1.0
    # walk
    val = 2; a[x == val] = 1.0; r[x == val] = 0.01; g[x == val] = 0.01; b[x == val] = 0.01
    # gt
    val = 4; a[x == val] = 1.0; r[x == val] = 1.0; g[x == val] = 0.0; b[x == val] = 0.0
    # cd_rl
    val = 5; a[x == val] = 1.0; r[x == val] = 31/255.0; g[x == val] = 151/255.0; b[x == val] = 1.0
    # cd_heat
    val = 6; a[x == val] = 1.0; r[x == val] = 0.0; g[x == val] = 1.0; b[x == val] = 0.0
    # cd_rule
    val = 7; a[x == val] = 1.0; r[x == val] = 1.0; g[x == val] = 0.06; b[x == val] = 0.94

    return r, g, b, a


def render(datacube, walk_mask, xyzangles=[0, 0, 0], save_filename="volumerender"+str(0)+".png"):
    print("calculating camera grid...")

    datacube = np.float32(datacube)
    datacube = nimg.uniform_filter(datacube, size=5)
    #datacube[datacube>0] = 1

    sx = nimg.sobel(datacube, axis=0, mode='constant')
    sy = nimg.sobel(datacube, axis=1, mode='constant')
    sz = nimg.sobel(datacube, axis=2, mode='constant')
    sob = np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)
    datacube = np.float32(sob)
    datacube /= datacube.max()

    # Datacube Grid
    Nx, Ny, Nz = datacube.shape
    x = np.linspace(-Nx / 2, Nx / 2, Nx)
    y = np.linspace(-Ny / 2, Ny / 2, Ny)
    z = np.linspace(-Nz / 2, Nz / 2, Nz)
    points = (x, y, z)

    # Camera Grid / Query Points -- rotate camera view
    N = datacube.shape[0]
    M = 1*N
    c = np.linspace(-N / 2, N / 2, M)
    qx, qy, qz = np.meshgrid(c, c, c)

    rot = R.from_euler('YZX', xyzangles, degrees=True)
    q = np.array([qx.ravel(), qy.ravel(), qz.ravel()]).T
    qi = rot.apply(q)

    # ==== FIRST, RENDER VOLUME ONLY ()
    datacube2 = datacube.copy()
    #datacube2[datacube2>1] = 1

    camera_grid = interpn(points, datacube2, qi, method='linear', bounds_error=False, fill_value=0.0).reshape((M, M, M))

    grid = camera_grid.copy()
    #grid[grid>1] = 0
    print("computing surface-norms...")
    # get gradients/surface-norms
    grads = np.gradient(nimg.uniform_filter(grid, size=5))

    grads_magnitude = np.sqrt(grads[0]**2+grads[1]**2+grads[2]**2)
    grads_magnitude[grads_magnitude<1e-3] = 1
    grads_norm = grads[0]/grads_magnitude, grads[1]/grads_magnitude, grads[2]/grads_magnitude

    datacube2[walk_mask > 3] = walk_mask[walk_mask > 3]
    camera_grid = interpn(points, datacube2, qi, method='nearest', bounds_error=False, fill_value=0.0).reshape((M, M, M))
    camera_grid = np.round(camera_grid)

    # Interpolate onto Camera Grid
    #print(datacube.shape, qi.shape)

    print("determining colors and shades...")
    # color and shading
    red, green, blue, alpha = myTransferFunction(camera_grid)
    alpha[:] = grid[:]*c_alpha_mask
    alpha[camera_grid>2] = 1.0
    light_dir = np.array([1, 0.0, 0.0])
    light_dir = light_dir/np.linalg.norm(light_dir)

    grads_norm = grads_norm[0]*light_dir[0] + grads_norm[1]*light_dir[1] + grads_norm[2]*light_dir[2]
    #grads_norm[grads_norm<0.5] = 0.5
    grads_norm[camera_grid>2] = 7.0



    red = np.clip(red*grads_norm, 0.0, 1.0)
    green = np.clip(green * grads_norm, 0.0, 1.0)
    blue = np.clip(blue * grads_norm, 0.0, 1.0)
    alpha = np.clip(alpha * grads_norm, 0.005, 1.0)

    print("compositing...")
    # Do Volume Rendering
    image = np.ones((camera_grid.shape[1], camera_grid.shape[2], 3))*0.0
    for z, dataslice in enumerate(camera_grid):
        r, g, b, a = red[z, :, :], green[z, :, :], blue[z, :, :], alpha[z, :, :]
        image[:, :, 0] = a * r + (1 - a) * image[:, :, 0]
        image[:, :, 1] = a * g + (1 - a) * image[:, :, 1]
        image[:, :, 2] = a * b + (1 - a) * image[:, :, 2]

    image = nimg.median_filter(image, 3)
    image = np.clip(image, 0.0, 1.0)

    plt.figure(figsize=(4, 4), dpi=80)
    plt.imshow(image)
    plt.axis('off')

    plt.show()
    plt.pause(0.05)
    plt.savefig(save_filename, dpi=240, bbox_inches='tight', pad_inches=0)


def crop_datacube(datacube, use_preset=False, margin=50):
    global p
    if not use_preset:
        p = np.argwhere(datacube>0)

    p_min = np.min(p, axis=0)
    p_max = np.max(p, axis=0)

    p_min -= margin
    p_max += margin
    p_min = np.clip(p_min, 0, datacube.shape)
    p_max = np.clip(p_max, 0, datacube.shape)
    return datacube[p_min[0]:p_max[0], p_min[1]:p_max[1], p_min[2]:p_max[2]]


def get_key(args, key):
    val = None
    try:
        val = args[key]
    except:
        pass
    return val


def fetch_args():
    args = {}
    for k, arg in enumerate(sys.argv):
        if arg[0]=='-':
            args[arg[1:]] = sys.argv[k+1]
    return args


def process_args(args):
    global volume_path, mask_path, xyzangles
    global save_filename
    volume_path = get_key(args, "volume_path")
    mask_path = get_key(args, "mask_path")
    xyzangles = get_key(args, "xyzangles")
    save_filename = get_key(args, "save_filename")
    if xyzangles is None:
        xyzangles = "[0,-20,30]"
    xyzangles = xyzangles.strip("[]").split(',')
    xyzangles = [int(x) for x in xyzangles]


def main():
    global save_filename
    args = fetch_args()
    process_args(args)

    if volume_path is not None:
        datacube = io.loadmat(volume_path)
        key = None
        for k in datacube.keys():
            if k[0] != "_":
                key = k
                break
        datacube = datacube[key]

    mask = None
    if mask_path is not None:
        mask = io.loadmat(mask_path)
        key = None
        for k in mask.keys():
            if k[0]!="_":
                key = k
                break
        mask = mask[key]

    if save_filename is None:
        save_filename = 'render.png'

    render(datacube, mask, xyzangles=xyzangles, save_filename=save_filename)


if __name__ == '__main__':
    main()

"""
import numpy as np
import scipy.io as io
a = np.ones([200,200,200])*0.0
a[50:150, 50:150, 100] = 4.0

io.savemat('mask.mat', {'m': a})

"""