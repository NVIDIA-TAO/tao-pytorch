# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyCUDA implementation of IoU of rotated boxes."""
import numpy as np
import pycuda # noqa pylint: disable=W0611
import pycuda.autoinit # noqa pylint: disable=W0611
from pycuda.compiler import SourceModule
import pycuda.driver as cuda


pyc_dev = pycuda.autoinit.device
pyc_ctx = pyc_dev.retain_primary_context()


mod = SourceModule("""
    //PyCUDA implementation of IoU of rotated boxes. This is basically a C++/CUDA
    // translation of the original numpy/numba based implementation.
    // area of trangle by cross product: S = |ca| * |cb| * sin(theta)/2 = ||ca x cb||/2
    __device__ float trangle_area(float *a, float *b, float *c)
    {
        return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) *
                (b[0] - c[0])) / 2.0;
    }
    // area of polygons as the sum of areas of triangles
    __device__ float area(float *int_pts, int num_of_inter)
    {
        float area_val = 0.0;
        for (int i=0; i<num_of_inter - 2; i++)
        {
            area_val += fabsf(
                trangle_area(   int_pts, //int_pts[:2],
                                &int_pts[2 * i + 2],
                                &int_pts[2 * i + 4]));
        }
        return area_val;
    }
    // sort the vertices in a convex polygon
    __device__ void sort_vertex_in_convex_polygon(float *int_pts, int num_of_inter)
    {
        if (num_of_inter > 0)
        {
            float center[2] ={0.0,0.0};
            //center[:] = 0.0
            for (int i=0; i<num_of_inter; i++)
            {
                center[0] += int_pts[2 * i];
                center[1] += int_pts[2 * i + 1];
            }
            center[0] /= num_of_inter;
            center[1] /= num_of_inter;
            float v[2];
            float vs[16];
            for (int i=0; i<num_of_inter; i++)
            {
                v[0] = int_pts[2 * i] - center[0];
                v[1] = int_pts[2 * i + 1] - center[1];
                float d = sqrtf(v[0] * v[0] + v[1] * v[1]);
                v[0] = v[0] / d;
                v[1] = v[1] / d;
                if (v[1] < 0)
                    v[0] = -2 - v[0];
                vs[i] = v[0];
            }
            int j = 0;
            float temp = 0;
            for (int i=0; i<num_of_inter; i++)
            {
                if (vs[i - 1] > vs[i])
                {
                    temp = vs[i];
                    float tx = int_pts[2 * i];
                    float ty = int_pts[2 * i + 1];
                    j = i;
                    while (j > 0 && (vs[j - 1] > temp) )
                    {
                        vs[j] = vs[j - 1];
                        int_pts[j * 2] = int_pts[j * 2 - 2];
                        int_pts[j * 2 + 1] = int_pts[j * 2 - 1];
                        j -= 1;
                    }
                    vs[j] = temp;
                    int_pts[j * 2] = tx;
                    int_pts[j * 2 + 1] = ty;
                }
            }
        }
    }
    // intersection of two line segments
    __device__  int line_segment_intersection(float *pts1, float *pts2, int i, int j, float *temp_pts)
    {
        float A[2];
        float B[2];
        float C[2];
        float D[2];
        A[0] = pts1[2 * i];
        A[1] = pts1[2 * i + 1];
        B[0] = pts1[2 * ((i + 1) % 4)];
        B[1] = pts1[2 * ((i + 1) % 4) + 1];
        C[0] = pts2[2 * j];
        C[1] = pts2[2 * j + 1];
        D[0] = pts2[2 * ((j + 1) % 4)];
        D[1] = pts2[2 * ((j + 1) % 4) + 1];
        float BA0 = B[0] - A[0];
        float BA1 = B[1] - A[1];
        float DA0 = D[0] - A[0];
        float CA0 = C[0] - A[0];
        float DA1 = D[1] - A[1];
        float CA1 = C[1] - A[1];
        float acd = DA1 * CA0 > CA1 * DA0;
        float bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0]);
        if (acd != bcd)
        {
            float abc = CA1 * BA0 > BA1 * CA0;
            float abd = DA1 * BA0 > BA1 * DA0;
            if (abc != abd)
            {
                float DC0 = D[0] - C[0];
                float DC1 = D[1] - C[1];
                float ABBA = A[0] * B[1] - B[0] * A[1];
                float CDDC = C[0] * D[1] - D[0] * C[1];
                float DH = BA1 * DC0 - BA0 * DC1;
                float Dx = ABBA * DC0 - BA0 * CDDC;
                float Dy = ABBA * DC1 - BA1 * CDDC;
                temp_pts[0] = Dx / DH;
                temp_pts[1] = Dy / DH;
                return 1;
            }
        }
        return 0;
    }
    // whether point q is on line segment of (p1, p2) and in between the 2 points on the line
    __device__ int on_segment(float p1_x, float p1_y, float p2_x, float p2_y, float q_x, float q_y)
    {
        return (
            ( (q_x - p1_x) * (p2_y - p1_y) == (p2_x - p1_x) * (q_y - p1_y) ) &&
            ( min(p1_x, p2_x) <= q_x )  &&
            ( q_x <= max(p1_x, p2_x) )  &&
            ( min(p1_y, p2_y) <= q_y )  &&
            ( q_y <= max(p1_y, p2_y) )
        );
    }
    // whether a point is in a quadrilateral
    __device__ int in_quadrilateral(float pt_x, float pt_y, float *corners)
    {
        int flag = 0;
        int j=0;
        float a_x, a_y, b_x, b_y;
        for (int i=0; i<4; i++)
        {
            j = (i + 1) % 4;
            a_x = corners[2 * i];
            a_y = corners[2 * i + 1];
            b_x = corners[2 * j];
            b_y = corners[2 * j + 1];
            if (on_segment(a_x, a_y, b_x, b_y, pt_x, pt_y))
                return 1;
            if (
                (((a_y - pt_y) > 0) != ((b_y - pt_y) > 0)) &&
                pt_x - (pt_y - a_y) * (a_x - b_x) / (a_y - b_y) - a_x < 0
            )
                flag = ! flag;
        }
        return flag;
    }
    // intersection of 2 quadrilaterals
    __device__ int quadrilateral_intersection(float *pts1, float *pts2, float *int_pts)
    {
        int num_of_inter = 0;
        float temp_pts[2];
        int has_pts;
        for (int i=0; i< 4; i++)
        {
            if ( in_quadrilateral(pts1[2 * i], pts1[2 * i + 1], pts2) )
            {
                int_pts[num_of_inter * 2] = pts1[2 * i];
                int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1];
                num_of_inter += 1;
            }
            if ( in_quadrilateral(pts2[2 * i], pts2[2 * i + 1], pts1) )
            {
                int_pts[num_of_inter * 2] = pts2[2 * i];
                int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1];
                num_of_inter += 1;
            }
        }
        for (int i=0; i<4; i++)
        {
            for (int j=0; j<4; j++)
            {
                int has_pts = line_segment_intersection(pts1, pts2, i, j, temp_pts);
                if (has_pts)
                {
                    int_pts[num_of_inter * 2] = temp_pts[0];
                    int_pts[num_of_inter * 2 + 1] = temp_pts[1];
                    num_of_inter += 1;
                }
            }
        }
        return num_of_inter;
    }
    // convert rotated boxes to corners format
    __device__ void rbbox_to_corners(float *corners, float *rbbox)
    {
        float angle = rbbox[4];
        float a_cos = cosf(angle);
        float a_sin = sinf(angle);
        float center_x = rbbox[0];
        float center_y = rbbox[1];
        float x_d = rbbox[2];
        float y_d = rbbox[3];
        float corners_x[4];
        float corners_y[4];
        corners_x[0] = -x_d / 2;
        corners_x[1] = -x_d / 2;
        corners_x[2] = x_d / 2;
        corners_x[3] = x_d / 2;
        corners_y[0] = -y_d / 2;
        corners_y[1] = y_d / 2;
        corners_y[2] = y_d / 2;
        corners_y[3] = -y_d / 2;
        for (int i=0; i<4; i++)
        {
            corners[2 *
                    i] = a_cos * corners_x[i] + a_sin * corners_y[i] + center_x;
            corners[2 * i
                    + 1] = -a_sin * corners_x[i] + a_cos * corners_y[i] + center_y;
        }
    }
    // intersection of 2 rorated boxes
    __device__ float inter(float *rbbox1, float *rbbox2)
    {
        float corners1[8];
        float corners2[8];
        float intersection_corners[16];
        int num_intersection;
        rbbox_to_corners(corners1, rbbox1);
        rbbox_to_corners(corners2, rbbox2);
        num_intersection = quadrilateral_intersection(corners1, corners2,
                                                      intersection_corners);
        sort_vertex_in_convex_polygon(intersection_corners, num_intersection);
        return area(intersection_corners, num_intersection);
    }
    // compute IoU of a pair of rotated boxes
    __device__ float devRotateIoUEval(float *rbox1, float *rbox2, int criterion)
    {
        //centerx, centery, widthx, heighty, angle
        float area1 = rbox1[2] * rbox1[3];
        float area2 = rbox2[2] * rbox2[3];
        float area_inter = inter(rbox1, rbox2);
        if (criterion == -1)
            return area_inter / (area1 + area2 - area_inter);
        if (criterion == 0)
            return area_inter / area1;
        if (criterion == 1)
            return area_inter / area2;
        return area_inter;
    }
    // CUDA kernel to compute IoU of multiple rotated boxes
    __global__ void rotate_iou_gpu_eval(float *q, float *b, int q_max, int b_max, float *output, int criterion=-1)
    {
        const int block_size=64;
        const int point_len = 5;
        extern __shared__ float block_boxes[block_size*point_len];
        extern __shared__ float block_qboxes[block_size*point_len];
        int block_row_index=blockIdx.x;
        int block_col_index=blockIdx.y;
        int tx=threadIdx.x;
        float *b_addr_this_block=b + block_row_index * block_size * point_len;
        float *q_addr_this_block=q + block_col_index * block_size * point_len;
        int b_valid_len_this_block = b_max - block_row_index*block_size < block_size ? (b_max - block_row_index*block_size) : block_size;
        int q_valid_len_this_block = q_max - block_col_index*block_size < block_size ? (q_max - block_col_index*block_size) : block_size;
        if (tx < b_valid_len_this_block)
        {
            for (int i=0; i<point_len; i++)
            {
                block_boxes[tx * point_len + i] = b_addr_this_block[tx * point_len + i];
            }
        }
        if (tx < q_valid_len_this_block)
        {
            for (int i=0; i<point_len; i++)
            {
                block_qboxes[tx * point_len + i] = q_addr_this_block[tx * point_len + i];
            }
        }
        __syncthreads();
        if (tx<b_valid_len_this_block)
        {
            int rows_index=block_row_index * block_size + tx;
            int cols_index=block_col_index * block_size;
            for (int i=0;i<q_valid_len_this_block;i++)
            {
                (output + rows_index* q_max + cols_index + i)[0] = devRotateIoUEval(&block_boxes[point_len*tx], &block_qboxes[point_len * i], criterion);
            }
        }
    }
  """)


def div_up(m, n):
    """Division with round to up."""
    return m // n + (m % n > 0)


def rotate_iou_gpu_eval(box_np, query_np, criterion=-1):
    """IoU of rotated boxes."""
    N = box_np.shape[0]
    K = query_np.shape[0]
    if N == 0 or K == 0:
        return np.zeros((N, K), dtype=np.float32)
    pyc_ctx.push()
    box_np = box_np.astype(np.float32, order='C')
    query_np = query_np.astype(np.float32, order='C')
    iou_np = np.zeros((N, K), dtype=np.float32, order='C')
    threadsPerBlock = 8 * 8
    blockspergrid = (div_up(N, threadsPerBlock), div_up(K, threadsPerBlock))
    func = mod.get_function("rotate_iou_gpu_eval")
    func(
        cuda.In(query_np),
        cuda.In(box_np),
        np.int32(K),
        np.int32(N),
        cuda.Out(iou_np),
        np.int32(criterion),
        grid=blockspergrid,
        block=(threadsPerBlock, 1, 1)
    )
    pyc_ctx.pop()
    return iou_np
