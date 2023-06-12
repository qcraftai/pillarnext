import numba
import numpy as np


def flip(boxes, axis):
    if axis == 'x':
        boxes[:, 1] = -boxes[:, 1]
        boxes[:, -1] = -boxes[:, -1]
        if boxes.shape[1] > 7:  # x, y, z, l, w, h, vx, vy, rz
            boxes[:, 7] = -boxes[:, 7]
    elif axis == 'y':
        boxes[:, 0] = -boxes[:, 0]
        boxes[:, -1] = -boxes[:, -1] + np.pi

        if boxes.shape[1] > 7:
            boxes[:, 6] = -boxes[:, 6]
    else:
        raise Exception('Unknown flip axis!')

    cond = boxes[:, -1] > np.pi
    boxes[cond, -1] = boxes[cond, -1] - 2 * np.pi

    cond = boxes[:, -1] < -np.pi
    boxes[cond, -1] = boxes[cond, -1] + 2 * np.pi

    return boxes


def scaling(boxes, noise):
    boxes[:, :-1] *= noise
    return boxes


def rotate(boxes, noise_rotation):
    boxes[:, :3] = yaw_rotation(boxes[:, :3], noise_rotation)
    if boxes.shape[1] > 7:
        boxes[:, 6:8] = yaw_rotation(np.hstack([boxes[:, 6:8], np.zeros((boxes.shape[0], 1))]),
                                     noise_rotation)[:, :2]

    boxes[:, -1] += noise_rotation
    return boxes


def translate(boxes, noise_translate):
    boxes[:, :3] += noise_translate
    return boxes


def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1
    ).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * \
        corners_norm.reshape([1, 2 ** ndim, ndim])
    return corners


def center_to_corner_box3d(boxes):
    """convert locations, dimensions and angles to corners

    Args:
        centers (float array, shape=[N, 3]): x, y, z
        dims (float array, shape=[N, 3]): size_x, size_y, size_z
        angles (float array, shape=[N]): yaw angle
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    corners = corners_nd(boxes[:, 3:6])
    # corners: [N, 8, 3]
    corners = rotation_3d(corners, boxes[:, -1])
    corners += boxes[:, :3].reshape([-1, 1, 3])
    return corners


def center_to_corner_box2d(boxes):  # corrected
    """convert locations, dimensions and angles to corners.
    format: center(xy), dims(xy), yaw_angle

    Args:
        centers (float array, shape=[N, 2]): x, y
        dims (float array, shape=[N, 2]): size_x, size_y,
        angles (float array, shape=[N]): yaw angle

    Returns:
        [type]: [description]
    """
    corners = corners_nd(boxes[:, 2:4])
    # corners: [N, 4, 2]
    corners = rotation_2d(corners, boxes[:, -1])
    corners += boxes[:, :2].reshape([-1, 1, 2])
    return corners


def yaw_rotation(points, yaw):
    rot_sin = np.sin(yaw)
    rot_cos = np.cos(yaw)

    rot_mat_T = np.stack(
        [
            [rot_cos, rot_sin, 0],
            [-rot_sin, rot_cos, 0],
            [0, 0, 1],
        ]
    )

    return points @ rot_mat_T


def rotation_3d(points, yaw):  # corrected
    # points: [N, 8, 3]
    rot_sin = np.sin(yaw)
    rot_cos = np.cos(yaw)
    zeros = np.zeros_like(rot_sin)
    ones = np.ones_like(rot_sin)

    rot_mat_T = np.stack(
        [
            [rot_cos, rot_sin, zeros],
            [-rot_sin, rot_cos, zeros],
            [zeros, zeros, ones],
        ]
    )

    return np.einsum("aij,jka->aik", points, rot_mat_T)


def rotation_2d(points, angles):  # corrected, counterclock wise
    """rotation 2d points with given angle

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_T = np.stack([[rot_cos, rot_sin], [-rot_sin, rot_cos]])
    return np.einsum("aij,jka->aik", points, rot_mat_T)


@numba.njit
def corner_to_standup_nd_jit(boxes_corner):
    num_boxes = boxes_corner.shape[0]
    ndim = boxes_corner.shape[-1]
    result = np.zeros((num_boxes, ndim * 2), dtype=boxes_corner.dtype)
    for i in range(num_boxes):
        for j in range(ndim):
            result[i, j] = np.min(boxes_corner[i, :, j])
        for j in range(ndim):
            result[i, j + ndim] = np.max(boxes_corner[i, :, j])
    return result


def points_in_rbbox(points, boxes):
    indices = np.zeros((points.shape[0], boxes.shape[0]), dtype=bool)
    points_in_boxes_jit(points, boxes, indices)
    return indices


@numba.njit
def points_in_boxes_jit(points, boxes, indices):
    '''
    Input:
        points: float array [N, *],
        boxes:  float array[M, 7] or [M, 9],
                with first 6 dimensions x, y, z, length, width, height, last dimension yaw angle
    return:
        bool array of shape [N, M]
    '''
    num_points = points.shape[0]
    num_boxes = boxes.shape[0]
    for j in range(num_boxes):
        for i in range(num_points):
            if np.abs(points[i, 2] - boxes[j, 2]) <= boxes[j, 5] / 2.0:
                cosa = np.cos(boxes[j, -1])
                sina = np.sin(boxes[j, -1])
                shift_x = points[i, 0] - boxes[j, 0]
                shift_y = points[i, 1] - boxes[j, 1]
                local_x = shift_x * cosa + shift_y * sina
                local_y = -shift_x * sina + shift_y * cosa
                indices[i, j] = np.logical_and(np.abs(local_x) <= boxes[j, 3] / 2.0,
                                               np.abs(local_y) <= boxes[j, 4] / 2.0)


@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack(
        (boxes, boxes[:, slices, :]), axis=2
    )  # [N, 4, 2(line), 2(xy)]
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    # vec = np.zeros((2,), dtype=boxes.dtype)
    boxes_standup = corner_to_standup_nd_jit(boxes)
    qboxes_standup = corner_to_standup_nd_jit(qboxes)
    for i in range(N):
        for j in range(K):
            # calculate standup first
            iw = min(boxes_standup[i, 2], qboxes_standup[j, 2]) - max(
                boxes_standup[i, 0], qboxes_standup[j, 0]
            )
            if iw > 0:
                ih = min(boxes_standup[i, 3], qboxes_standup[j, 3]) - max(
                    boxes_standup[i, 1], qboxes_standup[j, 1]
                )
                if ih > 0:
                    for k in range(4):
                        for l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, l, 0]
                            D = lines_qboxes[j, l, 1]
                            acd = (D[1] - A[1]) * (C[0] - A[0]) > (C[1] - A[1]) * (
                                D[0] - A[0]
                            )
                            bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (
                                D[0] - B[0]
                            )
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (
                                    C[0] - A[0]
                                )
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (
                                    D[0] - A[0]
                                )
                                if abc != abd:
                                    ret[i, j] = True  # collision.
                                    break
                        if ret[i, j] is True:
                            break
                    if ret[i, j] is False:
                        # now check complete overlap.
                        # box overlap qbox:
                        box_overlap_qbox = True
                        for l in range(4):  # point l in qboxes
                            for k in range(4):  # corner k in boxes
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * \
                                    (boxes[i, k, 0] - qboxes[j, l, 0])
                                cross -= vec[0] * \
                                    (boxes[i, k, 1] - qboxes[j, l, 1])
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if box_overlap_qbox is False:
                                break

                        if box_overlap_qbox is False:
                            qbox_overlap_box = True
                            for l in range(4):  # point l in boxes
                                for k in range(4):  # corner k in qboxes
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * \
                                        (qboxes[j, k, 0] - boxes[i, l, 0])
                                    cross -= vec[0] * \
                                        (qboxes[j, k, 1] - boxes[i, l, 1])
                                    if cross >= 0:  #
                                        qbox_overlap_box = False
                                        break
                                if qbox_overlap_box is False:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True  # collision.
                        else:
                            ret[i, j] = True  # collision.
    return ret
