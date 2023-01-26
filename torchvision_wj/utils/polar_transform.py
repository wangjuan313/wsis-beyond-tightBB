import torch
import numpy as np


def bilinear_interpolate(image, coords):
    if image.ndim == 2:
        x, y = coords[0], coords[1]
    elif image.ndim == 3:
        x, y, z = coords[0], coords[1], coords[2]
        z = z.long()
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1
    
    x0 = torch.clamp(x0, 0, image.shape[1]-1)
    x1 = torch.clamp(x1, 0, image.shape[1]-1)
    y0 = torch.clamp(y0, 0, image.shape[0]-1)
    y1 = torch.clamp(y1, 0, image.shape[0]-1)

    if image.ndim == 2:
        Ia = image[y0, x0]
        Ib = image[y1, x0]
        Ic = image[y0, x1]
        Id = image[y1, x1]
    elif image.ndim == 3:
        Ia = image[y0, x0, z]
        Ib = image[y1, x0, z]
        Ic = image[y0, x1, z]
        Id = image[y1, x1, z]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return Ia*wa + Ib*wb + Ic*wc + Id*wd


def _stackcopy(a, b):
    """Copy b into each color layer of a, such that::
      a[:,:,0] = a[:,:,1] = ... = b
    Parameters
    ----------
    a : (M, N) or (M, N, P) ndarray
        Target array.
    b : (M, N)
        Source array.
    Notes
    -----
    Color images are stored as an ``(M, N, 3)`` or ``(M, N, 4)`` arrays.
    """
    if a.ndim == 3:
        a[:] = b[:, :, np.newaxis]
    else:
        a[:] = b
        

def _linear_polar_mapping(output_coords, k_angle, k_radius, center):
    """Inverse mapping function to convert from cartesian to polar coordinates
    Parameters
    ----------
    output_coords : ndarray
        `(M, 2)` array of `(col, row)` coordinates in the output image
    k_angle : float
        Scaling factor that relates the intended number of rows in the output
        image to angle: ``k_angle = nrows / (2 * np.pi)``
    k_radius : float
        Scaling factor that relates the radius of the circle bounding the
        area to be transformed to the intended number of columns in the output
        image: ``k_radius = ncols / radius``
    center : tuple (row, col)
        Coordinates that represent the center of the circle that bounds the
        area to be transformed in an input image.
    Returns
    -------
    coords : ndarray
        `(M, 2)` array of `(col, row)` coordinates in the input image that
        correspond to the `output_coords` given as input.
    """
    angle = output_coords[:, 1] / k_angle
    rr = ((output_coords[:, 0] / k_radius) * torch.sin(angle)) + center[1]
    cc = ((output_coords[:, 0] / k_radius) * torch.cos(angle)) + center[0]    
    coords = torch.stack((cc, rr), dim=1)
    return coords


def _log_polar_mapping(output_coords, k_angle, k_radius, center):
    """Inverse mapping function to convert from cartesian to polar coordinates
    Parameters
    ----------
    output_coords : ndarray
        `(M, 2)` array of `(col, row)` coordinates in the output image
    k_angle : float
        Scaling factor that relates the intended number of rows in the output
        image to angle: ``k_angle = nrows / (2 * np.pi)``
    k_radius : float
        Scaling factor that relates the radius of the circle bounding the
        area to be transformed to the intended number of columns in the output
        image: ``k_radius = width / np.log(radius)``
    center : tuple (row, col)
        Coordinates that represent the center of the circle that bounds the
        area to be transformed in an input image.
    Returns
    -------
    coords : ndarray
        `(M, 2)` array of `(col, row)` coordinates in the input image that
        correspond to the `output_coords` given as input.
    """
    angle = output_coords[:, 1] / k_angle
    rr = ((torch.exp(output_coords[:, 0] / k_radius)) * torch.sin(angle)) + center[1]
    cc = ((torch.exp(output_coords[:, 0] / k_radius)) * torch.cos(angle)) + center[0]
    coords = torch.stack((cc, rr), dim=1)
    return coords
        

def polar_transform(image, center=None, radius=None, output_shape=None,
                    height=360, scaling='linear'):
    
    device = image.device
    dtype  = image.dtype
    if center is None:
        center = (torch.tensor(image.shape, dtype=dtype)[:2] / 2) - 0.5
    center = center.to(device)
    # radius = torch.tensor(radius)

    if output_shape is None:
        width = int(np.ceil(radius))
        output_shape = (height, width)
    else:
        height = output_shape[0]
        width = output_shape[1]

    if scaling == 'linear':
        k_radius = width / radius
        map_func = _linear_polar_mapping
    elif scaling == 'log':
        k_radius = width / torch.log(radius)
        map_func = _log_polar_mapping
    else:
        raise ValueError("Scaling value must be in {'linear', 'log'}")
    k_angle = height / (2 * np.pi)
    
    input_shape = image.shape
    if len(input_shape) == 3 and len(output_shape) == 2:
        output_shape = (output_shape[0], output_shape[1], input_shape[2])
    rows, cols = output_shape[0], output_shape[1]
    coords_shape = [len(output_shape), rows, cols]
    if len(output_shape) == 3:
        coords_shape.append(output_shape[2])
    coords = image.new_full(coords_shape, 0, dtype=dtype)
    # Reshape grid coordinates into a (P, 2) array of (row, col) pairs
    # tf_coords = np.indices((cols, rows), dtype=dtype).reshape(2, -1).T
    shifts_y = torch.arange(0, rows, dtype=torch.float32, device=device)
    shifts_x = torch.arange(0, cols, dtype=torch.float32, device=device)
    tf_coords = torch.meshgrid(shifts_x, shifts_y)
    tf_coords = torch.stack(tf_coords, dim=0).reshape(2, -1).T

    # Map each (row, col) pair to the source image according to
    # the user-provided mapping
    tf_coords = map_func(tf_coords, k_angle=k_angle, k_radius=k_radius, center=center)
    # Reshape back to a (2, M, N) coordinate grid
    tf_coords = tf_coords.T.reshape((-1, cols, rows)).permute(0, 2, 1)
    
    # Place the y-coordinate mapping
    _stackcopy(coords[1, ...], tf_coords[0, ...])
    # Place the x-coordinate mapping
    _stackcopy(coords[0, ...], tf_coords[1, ...])

    if len(output_shape) == 3:
        coords[2, ...] = torch.arange(0,output_shape[2])

    warped = bilinear_interpolate(image, coords)

    # min_val = image.min()
    # max_val = image.max()
    # print('xxx', min_val, max_val)
    # warped = torch.clamp(warped, min_val, max_val)

    return warped