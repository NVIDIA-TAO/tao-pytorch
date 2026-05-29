# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Intersection Over Union (IoU) for 3D oriented bounding boxes."""

import numpy as np
from numpy.linalg import lstsq as optimizer
import scipy.spatial as sp
from scipy.spatial.transform import Rotation as rotation_util


_PLANE_THICKNESS_EPSILON = 0.000001
_POINT_IN_FRONT_OF_PLANE = 1
_POINT_ON_PLANE = 0
_POINT_BEHIND_PLANE = -1

EDGES = (
    [1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
    [1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
    [1, 2], [3, 4], [5, 6], [7, 8]  # lines along z-axis
)

BOTTOM = (
    [1, 6], [2, 5]
)

# The vertices are ordered according to the left-hand rule, so the normal
# vector of each face will point inward the box.
FACES = np.array([
    [5, 6, 8, 7],  # +x on yz plane
    [1, 3, 4, 2],  # -x on yz plane
    [3, 7, 8, 4],  # +y on xz plane = top
    [1, 2, 6, 5],  # -y on xz plane
    [2, 4, 8, 6],  # +z on xy plane = front
    [1, 5, 7, 3],  # -z on xy plane
])

UNIT_BOX = np.asarray([
    [0., 0., 0.],
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, 0.5, 0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, -0.5],
    [0.5, 0.5, 0.5],
])

NUM_KEYPOINTS = 9
FRONT_FACE_ID = 4
TOP_FACE_ID = 2


class Box(object):
    """General 3D Oriented Bounding Box."""

    def __init__(self, vertices=None):
        """Initialize a 3D bounding box"""
        if vertices is None:
            vertices = self.scaled_axis_aligned_vertices(np.array([1., 1., 1.]))

        self._vertices = vertices
        self._rotation = None
        self._translation = None
        self._scale = None
        self._transformation = None
        self._volume = None

    @classmethod
    def from_transformation(cls, rotation, translation, scale):
        """Constructs an oriented bounding box from transformation and scale.
        Args:
          rotation: a 3x3 rotation matrix.
          translation: a 3x1 translation vector.
          scale: A 3x1 vector, specifiying the size of the box in x-y-z dimension.
        """
        if rotation.size != 3 and rotation.size != 9:
            raise ValueError('Unsupported rotation, only 3x1 euler angles or 3x3 ' +
                             'rotation matrices are supported. ' + rotation)
        if rotation.size == 3:
            rotation = rotation_util.from_rotvec(rotation.tolist()).as_dcm()
        scaled_identity_box = cls.scaled_axis_aligned_vertices(scale)
        vertices = np.zeros((NUM_KEYPOINTS, 3))
        for i in range(NUM_KEYPOINTS):
            vertices[i, :] = np.matmul(
                rotation, scaled_identity_box[i, :]) + translation.flatten()
        return cls(vertices=vertices)

    def __repr__(self):
        """Built the representation of the 3D bounding box."""
        representation = 'Box: '
        for i in range(NUM_KEYPOINTS):
            representation += '[{0}: {1}, {2}, {3}]'.format(i, self.vertices[i, 0],
                                                            self.vertices[i, 1],
                                                            self.vertices[i, 2])
        return representation

    def __len__(self):
        """Return the number of keypoints"""
        return NUM_KEYPOINTS

    def __name__(self):
        """Return the name of the 3D bounding box"""
        return 'Box'

    def apply_transformation(self, transformation):
        """Applies transformation on the box.

        Group multiplication is the same as rotation concatenation. Therefore return
        new box with SE3(R * R2, T + R * T2); Where R2 and T2 are existing rotation
        and translation. Note we do not change the scale.

        Args:
          transformation: a 4x4 transformation matrix.

        Returns:
           transformed box.
        """
        if transformation.shape != (4, 4):
            raise ValueError('Transformation should be a 4x4 matrix.')

        new_rotation = np.matmul(transformation[:3, :3], self.rotation)
        new_translation = transformation[:3, 3] + (
            np.matmul(transformation[:3, :3], self.translation))
        return Box.from_transformation(new_rotation, new_translation, self.scale)

    @classmethod
    def scaled_axis_aligned_vertices(cls, scale):
        """Returns an axis-aligned set of verticies for a box of the given scale.

        Args:
          scale: A 3*1 vector, specifiying the size of the box in x-y-z dimension.
        """
        w = scale[0] / 2.
        h = scale[1] / 2.
        d = scale[2] / 2.

        # Define the local coordinate system, w.r.t. the center of the box
        aabb = np.array([[0., 0., 0.], [-w, -h, -d], [-w, -h, +d], [-w, +h, -d],
                         [-w, +h, +d], [+w, -h, -d], [+w, -h, +d], [+w, +h, -d],
                         [+w, +h, +d]])
        return aabb

    @classmethod
    def fit(cls, vertices):
        """Estimates a box 9-dof parameters from the given vertices.

        Directly computes the scale of the box, then solves for orientation and
        translation.

        Args:
          vertices: A 9*3 array of points. Points are arranged as 1 + 8 (center
            keypoint + 8 box vertices) matrix.

        Returns:
          orientation: 3*3 rotation matrix.
          translation: 3*1 translation vector.
          scale: 3*1 scale vector.
        """
        orientation = np.identity(3)
        translation = np.zeros((3, 1))
        scale = np.zeros(3)

        # The scale would remain invariant under rotation and translation.
        # We can safely estimate the scale from the oriented box.
        for axis in range(3):
            for edge_id in range(4):
                # The edges are stored in quadruples according to each axis
                begin, end = EDGES[axis * 4 + edge_id]
                scale[axis] += np.linalg.norm(vertices[begin, :] - vertices[end, :])
            scale[axis] /= 4.

        x = cls.scaled_axis_aligned_vertices(scale)
        system = np.concatenate((x, np.ones((NUM_KEYPOINTS, 1))), axis=1)
        solution, _, _, _ = optimizer(system, vertices, rcond=None)
        orientation = solution[:3, :3].T
        translation = solution[3, :3]
        return orientation, translation, scale

    def inside(self, point):
        """Tests whether a given point is inside the box.

          Brings the 3D point into the local coordinate of the box. In the local
          coordinate, the looks like an axis-aligned bounding box. Next checks if
          the box contains the point.
        Args:
          point: A 3*1 numpy vector.

        Returns:
          True if the point is inside the box, False otherwise.
        """
        inv_trans = np.linalg.inv(self.transformation)
        scale = self.scale
        point_w = np.matmul(inv_trans[:3, :3], point) + inv_trans[:3, 3]
        for i in range(3):
            if abs(point_w[i]) > scale[i] / 2.:
                return False
        return True

    def sample(self):
        """Samples a 3D point uniformly inside this box."""
        point = np.random.uniform(-0.5, 0.5, 3) * self.scale
        point = np.matmul(self.rotation, point) + self.translation
        return point

    @property
    def vertices(self):
        """Return the vertices of the 3D bounding box"""
        return self._vertices

    @property
    def rotation(self):
        """The rotation of the 3D bounding box"""
        if self._rotation is None:
            self._rotation, self._translation, self._scale = self.fit(self._vertices)
        return self._rotation

    @property
    def translation(self):
        """The translation of the 3D bounding box"""
        if self._translation is None:
            self._rotation, self._translation, self._scale = self.fit(self._vertices)
        return self._translation

    @property
    def scale(self):
        """The scale of the 3D bounding box"""
        if self._scale is None:
            self._rotation, self._translation, self._scale = self.fit(self._vertices)
        return self._scale

    @property
    def volume(self):
        """Compute the volume of the parallelpiped or the box.

          For the boxes, this is equivalent to np.prod(self.scale). However for
          parallelpiped, this is more involved. Viewing the box as a linear function
          we can estimate the volume using a determinant. This is equivalent to
          sp.ConvexHull(self._vertices).volume

        Returns:
          volume (float)
        """
        if self._volume is None:
            i = self._vertices[2, :] - self._vertices[1, :]
            j = self._vertices[3, :] - self._vertices[1, :]
            k = self._vertices[5, :] - self._vertices[1, :]
            sys = np.array([i, j, k])
            self._volume = abs(np.linalg.det(sys))
        return self._volume

    @property
    def transformation(self):
        """Built the transformation matrix using the translation and rotation"""
        if self._rotation is None:
            self._rotation, self._translation, self._scale = self.fit(self._vertices)
        if self._transformation is None:
            self._transformation = np.identity(4)
            self._transformation[:3, :3] = self._rotation
            self._transformation[:3, 3] = self._translation
        return self._transformation

    def get_ground_plane(self, gravity_axis=1):
        """Get ground plane under the box."""
        gravity = np.zeros(3)
        gravity[gravity_axis] = 1

        def get_face_normal(face, center):
            """Get a normal vector to the given face of the box."""
            v1 = self.vertices[face[0], :] - center
            v2 = self.vertices[face[1], :] - center
            normal = np.cross(v1, v2)
            return normal

        def get_face_center(face):
            """Get the center point of the face of the box."""
            center = np.zeros(3)
            for vertex in face:
                center += self.vertices[vertex, :]
            center /= len(face)
            return center

        ground_plane_id = 0
        ground_plane_error = 10.

        # The ground plane is defined as a plane aligned with gravity.
        # gravity is the (0, 1, 0) vector in the world coordinate system.
        for i in [0, 2, 4]:
            face = FACES[i, :]
            center = get_face_center(face)
            normal = get_face_normal(face, center)
            w = np.cross(gravity, normal)
            w_sq_norm = np.linalg.norm(w)
            if w_sq_norm < ground_plane_error:
                ground_plane_error = w_sq_norm
                ground_plane_id = i

        face = FACES[ground_plane_id, :]
        center = get_face_center(face)
        normal = get_face_normal(face, center)

        # For each face, we also have a parallel face that it's normal is also
        # aligned with gravity vector. We pick the face with lower height (y-value).
        # The parallel to face 0 is 1, face 2 is 3, and face 4 is 5.
        parallel_face_id = ground_plane_id + 1
        parallel_face = FACES[parallel_face_id]
        parallel_face_center = get_face_center(parallel_face)
        parallel_face_normal = get_face_normal(parallel_face, parallel_face_center)
        if parallel_face_center[gravity_axis] < center[gravity_axis]:
            center = parallel_face_center
            normal = parallel_face_normal
        return center, normal


class IoU(object):
    """General Intersection Over Union cost for Oriented 3D bounding boxes."""

    def __init__(self, box1, box2):
        """Initialize two 3D bounding boxes"""
        self._box1 = box1
        self._box2 = box2
        self._intersection_points = []

    def iou(self):
        """Computes the exact IoU using Sutherland-Hodgman algorithm."""
        self._intersection_points = []
        self._compute_intersection_points(self._box1, self._box2)
        self._compute_intersection_points(self._box2, self._box1)
        if self._intersection_points:
            try:
                intersection_volume = sp.ConvexHull(self._intersection_points).volume
                box1_volume = self._box1.volume
                box2_volume = self._box2.volume
                union_volume = box1_volume + box2_volume - intersection_volume
                return intersection_volume / union_volume
            except:  # noqa: E722
                return 0.
        else:
            return 0.

    def iou_sampling(self, num_samples=10000):
        """Computes intersection over union by sampling points.

        Generate n samples inside each box and check if those samples are inside
        the other box. Each box has a different volume, therefore the number o
        samples in box1 is estimating a different volume than box2. To address
        this issue, we normalize the iou estimation based on the ratio of the
        volume of the two boxes.

        Args:
          num_samples: Number of generated samples in each box

        Returns:
          IoU Estimate (float)
        """
        p1 = [self._box1.sample() for _ in range(num_samples)]
        p2 = [self._box2.sample() for _ in range(num_samples)]
        box1_volume = self._box1.volume
        box2_volume = self._box2.volume
        box1_intersection_estimate = 0
        box2_intersection_estimate = 0
        for point in p1:
            if self._box2.inside(point):
                box1_intersection_estimate += 1
        for point in p2:
            if self._box1.inside(point):
                box2_intersection_estimate += 1
        # We are counting the volume of intersection twice.
        intersection_volume_estimate = (box1_volume * box1_intersection_estimate +
                                        box2_volume * box2_intersection_estimate) / 2.0
        union_volume_estimate = (box1_volume * num_samples + box2_volume *
                                 num_samples) - intersection_volume_estimate
        iou_estimate = intersection_volume_estimate / union_volume_estimate
        return iou_estimate

    def _compute_intersection_points(self, box_src, box_template):
        """Computes the intersection of two boxes."""
        # Transform the source box to be axis-aligned
        inv_transform = np.linalg.inv(box_src.transformation)
        box_src_axis_aligned = box_src.apply_transformation(inv_transform)
        template_in_src_coord = box_template.apply_transformation(inv_transform)
        for face in range(len(FACES)):
            indices = FACES[face, :]
            poly = [template_in_src_coord.vertices[indices[i], :] for i in range(4)]
            clip = self.intersect_box_poly(box_src_axis_aligned, poly)
            for point in clip:
                # Transform the intersection point back to the world coordinate
                point_w = np.matmul(box_src.rotation, point) + box_src.translation
                self._intersection_points.append(point_w)

        for point_id in range(NUM_KEYPOINTS):
            v = template_in_src_coord.vertices[point_id, :]
            if box_src_axis_aligned.inside(v):
                point_w = np.matmul(box_src.rotation, v) + box_src.translation
                self._intersection_points.append(point_w)

    def intersect_box_poly(self, box, poly):
        """Clips the polygon against the faces of the axis-aligned box."""
        for axis in range(3):
            poly = self._clip_poly(poly, box.vertices[1, :], 1.0, axis)
            poly = self._clip_poly(poly, box.vertices[8, :], -1.0, axis)
        return poly

    def _clip_poly(self, poly, plane, normal, axis):
        """Clips the polygon with the plane using the Sutherland-Hodgman algorithm.

        See en.wikipedia.org/wiki/Sutherland-Hodgman_algorithm for the overview of
        the Sutherland-Hodgman algorithm. Here we adopted a robust implementation
        from "Real-Time Collision Detection", by Christer Ericson, page 370.

        Args:
          poly: List of 3D vertices defining the polygon.
          plane: The 3D vertices of the (2D) axis-aligned plane.
          normal: normal
          axis: A tuple defining a 2D axis.

        Returns:
          List of 3D vertices of the clipped polygon.
        """
        # The vertices of the clipped polygon are stored in the result list.
        result = []
        if len(poly) <= 1:
            return result

        # polygon is fully located on clipping plane
        poly_in_plane = True

        # Test all the edges in the polygon against the clipping plane.
        for i, current_poly_point in enumerate(poly):
            prev_poly_point = poly[(i + len(poly) - 1) % len(poly)]
            d1 = self._classify_point_to_plane(prev_poly_point, plane, normal, axis)
            d2 = self._classify_point_to_plane(current_poly_point, plane, normal,
                                               axis)
            if d2 == _POINT_BEHIND_PLANE:
                poly_in_plane = False
                if d1 == _POINT_IN_FRONT_OF_PLANE:
                    intersection = self._intersect(plane, prev_poly_point,
                                                   current_poly_point, axis)
                    result.append(intersection)
                elif d1 == _POINT_ON_PLANE:
                    if not result or (not np.array_equal(result[-1], prev_poly_point)):
                        result.append(prev_poly_point)
            elif d2 == _POINT_IN_FRONT_OF_PLANE:
                poly_in_plane = False
                if d1 == _POINT_BEHIND_PLANE:
                    intersection = self._intersect(plane, prev_poly_point,
                                                   current_poly_point, axis)
                    result.append(intersection)
                elif d1 == _POINT_ON_PLANE:
                    if not result or (not np.array_equal(result[-1], prev_poly_point)):
                        result.append(prev_poly_point)

                result.append(current_poly_point)
            else:
                if d1 != _POINT_ON_PLANE:
                    result.append(current_poly_point)

        return poly if poly_in_plane else result

    def _intersect(self, plane, prev_point, current_point, axis):
        """Computes the intersection of a line with an axis-aligned plane.

        Args:
          plane: Formulated as two 3D points on the plane.
          prev_point: The point on the edge of the line.
          current_point: The other end of the line.
          axis: A tuple defining a 2D axis.

        Returns:
          A 3D point intersection of the poly edge with the plane.
        """
        alpha = (current_point[axis] - plane[axis]) / (current_point[axis] - prev_point[axis])
        # Compute the intersecting points using linear interpolation (lerp)
        intersection_point = alpha * prev_point + (1.0 - alpha) * current_point
        return intersection_point

    def _inside(self, plane, point, axis):
        """Check whether a given point is on a 2D plane."""
        # Cross products to determine the side of the plane the point lie.
        x, y = axis
        u = plane[0] - point
        v = plane[1] - point

        a = u[x] * v[y]
        b = u[y] * v[x]
        return a >= b

    def _classify_point_to_plane(self, point, plane, normal, axis):
        """Classify position of a point w.r.t the given plane.

        See Real-Time Collision Detection, by Christer Ericson, page 364.

        Args:
          point: 3x1 vector indicating the point
          plane: 3x1 vector indicating a point on the plane
          normal: scalar (+1, or -1) indicating the normal to the vector
          axis: scalar (0, 1, or 2) indicating the xyz axis

        Returns:
          Side: which side of the plane the point is located.
        """
        signed_distance = normal * (point[axis] - plane[axis])
        position = None
        if signed_distance > _PLANE_THICKNESS_EPSILON:
            position = _POINT_IN_FRONT_OF_PLANE
        elif signed_distance < -_PLANE_THICKNESS_EPSILON:
            position = _POINT_BEHIND_PLANE
        else:
            position = _POINT_ON_PLANE
        return position

    @property
    def intersection_points(self):
        """Return the intersection points of the 3D bounding boxes"""
        return self._intersection_points
