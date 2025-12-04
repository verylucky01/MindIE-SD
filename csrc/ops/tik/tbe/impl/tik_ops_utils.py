# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from functools import partial

from constants import DTYPE_SIZE
from constants import FP16
from constants import FP32
from constants import L0C
from constants import UB


class TikOpsUtils:
    def __init__(self, tik_instance, repeat_once_size=128, max_repeat_times=255):
        self.tik_instance = tik_instance
        self.repeat_once_size = repeat_once_size
        self.block_size = 16
        self.max_repeat_times = max_repeat_times
        self.dtype = "float16"
        self.cont_data_mv_1_bust = partial(self.tik_instance.data_move, sid=0, nburst=1,
                                           src_stride=0,
                                           dst_stride=0)

    def MK_TO_K1MK0(self, mk_input_tensor, workspace_tensor=None):
        """change data shape from (M, K) to (K1, M, K0), K1 = K // K0, the effect is equant to:
        new_tensor =  np.stack(np.hsplit(mk_input_tensor, K1), axis=0)
        :param mk_input_tensor: input tensor in GM with shape: (M, K)
        :param workspace_tensor: workspace tensor with shape: (K1, M, K0)
        tensor will be changed, otherwise the new data will be copied to the workspace tensor,
        and input tensor will stay unchanged.
        :return: Tensor with shape (K1,M, K0)
        """
        dtype = mk_input_tensor.dtype
        m, k = mk_input_tensor.shape
        K0 = 16
        K1 = k // K0
        M = self.up_align_to_K0(m)
        if workspace_tensor is not None:
            with self.tik_instance.for_range(0, K1) as i:
                self.tik_instance.data_move(
                    workspace_tensor[i * M * K0:],
                    mk_input_tensor[i * K0:],
                    0,
                    M,
                    K0 * DTYPE_SIZE[dtype] // 32,
                    (K1 - 1) * K0 * DTYPE_SIZE[dtype] // 32,
                    0,
                )
            return workspace_tensor.reshape((K1, M, K0))
        else:
            with self.tik_instance.new_stmt_scope(disable_sync=False):
                tmp_ub = self.tik_instance.Tensor(dtype, (K1, M, K0), name="tmp_ub", scope=UB)
                # data_move(m,k) --> (k1,m,K0)
                with self.tik_instance.for_range(0, K1) as i:
                    self.tik_instance.data_move(
                        tmp_ub[i * M * K0:],
                        mk_input_tensor[i * K0:],
                        0,
                        M,
                        K0 * DTYPE_SIZE[dtype] // 32,
                        (K1 - 1) * K0 * DTYPE_SIZE[dtype] // 32,
                        0,
                    )
                self.cont_data_mv_1_bust(
                    dst=mk_input_tensor, src=tmp_ub, burst=K1 * M * K0 * DTYPE_SIZE[dtype] // 32)
                return mk_input_tensor.reshape((K1, M, K0))

    def transpose_matrix(self, src_ub, dst_ub, N, nk0=False):
        """ transpose matrix, default support shape: (16, n) -> (n, 16)
        if nk0 is true, support shape: (n, 16) -> (16, n)
        """
        K0 = 16
        rep_times = N // K0
        if nk0:
            src_list = [src_ub[16 * i] for i in range(16)]
            dst_list = [dst_ub[N * i] for i in range(16)]
        else:
            src_list = [src_ub[N * i] for i in range(16)]
            dst_list = [dst_ub[16 * i] for i in range(16)]

        dst_rep_stride = K0
        src_rep_stride = 1
        if rep_times == 1:
            dst_rep_stride = 0
            src_rep_stride = 0

        if nk0:
            src_rep_stride, dst_rep_stride = dst_rep_stride, src_rep_stride

        self.tik_instance.vec_trans_scatter(
            False, False, dst_list, src_list, rep_times, dst_rep_stride, src_rep_stride
        )
        return dst_ub

    def KN_TO_K1NK0(self, kn_input_tensor, workspace_tensor=None):
        """change data shape from (K,N) to (K1, N, K0), K1 = K // K0, the effect is equvilent to:
        new_tensor =  np.reshape(kn_input_tensor, newshape=(K1, K0, N)).swapaxes(1, 2)
        :param kn_input_tensor: input tensor with shape: (K, N)
        :param workspace_tensor: workspace tensor with shape: (K1, N, K0)
        tensor will be changed, otherwise the new data will be copied to the workspace tensor,
        and input tensor will stay unchanged.
        :return: Tensor with shape: (K1, N, K0)
        """
        dtype = kn_input_tensor.dtype
        k, n = kn_input_tensor.shape
        K0 = 16
        K1 = k // K0
        N = n
        with self.tik_instance.for_range(0, K1) as index:
            k1nk0_ub = self.tik_instance.Tensor(dtype, (N, K0), UB, "k1nk0_ub")
            src_ub = self.tik_instance.Tensor(dtype, (K0, N), UB, "src_ub")
            burst_len = K0 * N * DTYPE_SIZE[dtype] // 32
            self.cont_data_mv_1_bust(dst=src_ub, src=kn_input_tensor[index * K0 * N],
                                     burst=burst_len)
            k1nk0_ub = self.transpose_matrix(src_ub, k1nk0_ub, N)
            if workspace_tensor is None:
                self.cont_data_mv_1_bust(dst=kn_input_tensor[index * K0 * N], src=k1nk0_ub,
                                         burst=burst_len)
            else:
                self.cont_data_mv_1_bust(dst=workspace_tensor[index * K0 * N], src=k1nk0_ub,
                                         burst=burst_len)
        if workspace_tensor is None:
            return kn_input_tensor.reshape((K1, N, K0))
        else:
            return workspace_tensor.reshape((K1, N, K0))

    def N1MN0_TO_MN(self, N1MN0_input):
        """change data shape from (N1, M, N0) to (M, N), N0=16, N = N1 * K0, the effect is equant to:
        N1MN0_input = np.concatenate(list(map(np.squeeze, np.split(N1MN0_input, N1))), axis=1)
        :param N1MN0_input: input tensor with shape (N, M, N0) in GM or L1.
        :return:
        """
        dtype = N1MN0_input.dtype
        N1, M, N0 = N1MN0_input.shape

        with self.tik_instance.new_stmt_scope(disable_sync=False):
            tmp_ub = self.tik_instance.Tensor(dtype, (M, N1 * N0), name="tmp_ub", scope=UB)
            # data_move (n1,m,n0) --> (m,n)
            with self.tik_instance.for_range(0, N1) as i:
                self.tik_instance.data_move(
                    tmp_ub[i * N0:],
                    N1MN0_input[i * M * N0:],
                    0,
                    M,
                    N0 * DTYPE_SIZE[dtype] // 32,
                    0,
                    (N1 - 1) * N0 * DTYPE_SIZE[dtype] // 32,
                )
            # data_move out
            self.cont_data_mv_1_bust(dst=N1MN0_input, src=tmp_ub, burst=M * N1 * N0 * DTYPE_SIZE[
                dtype] // 32)
        return N1MN0_input.reshape((M, N1 * N0))

    def broadcast_vec_from_M_to_MN0(self, vec_ub):
        """broadcast a vec from (M,) to (M, 16)"""
        M = vec_ub.shape[0]
        dst_ub = self.tik_instance.Tensor(FP16, (M, 16), name="dst_ub", scope=UB)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            broadcast_l0c = self.tik_instance.Tensor(FP16, (16, M), name="broadcast_l0c", scope=L0C)
            self.tik_instance.broadcast_ub_to_l0c(broadcast_l0c, vec_ub, 1, M // 16, 1, 1)
            broadcast_ub = self.tik_instance.Tensor(FP16, (16, M), name="broadcast_ub", scope=UB)
            self.cont_data_mv_1_bust(dst=broadcast_ub, src=broadcast_l0c, burst=M * 16 * 2 // 512)
            # 将(M // N0, 16, N0)转换为(M, 16), 其中N0为16
            self.tik_instance.vec_trans(dst_ub, broadcast_ub, M // 16, 1, 1)
        return dst_ub

    def broadcast(self, vec_ub, shape):
        """ broadcast a vector to a matrix
        :param vec_ub: a tensor in UB with shape of (M,), and dtype is float16
        :param shape: the target shape, a tuple with value (M, N)，M and N are integer multiples of 16
        :return: a tensor in UB with shape of (M, N)
        """
        M, N = shape
        dst_ub = self.tik_instance.Tensor(FP16, shape, name="dst_ub", scope=UB)

        with self.tik_instance.new_stmt_scope(disable_sync=False):
            M_16_ub = self.broadcast_vec_from_M_to_MN0(vec_ub)
            # (M, 16) -> (M, 32) -> (M, 64) -> ... -> (M, N)
            self.tik_instance.data_move(dst_ub, M_16_ub, 0, M, 1, 0, N // 16 - 1)
            times = self.tik_instance.Scalar("int32", name="times", init_value=1)
            with self.tik_instance.for_range(begint=0, endt=N) as idx:
                offset = times * 16
                with self.tik_instance.if_scope(offset * 2 <= N):
                    burst = offset // 16
                    src_stride = N // 16 - burst
                    dst_stride = N // 16 - burst
                    self.tik_instance.data_move(dst_ub[offset], dst_ub, 0, M, burst, src_stride,
                                                dst_stride)
                with self.tik_instance.else_scope():
                    burst = (N - offset) // 16
                    src_stride = N // 16 - burst
                    dst_stride = N // 16 - burst
                    with self.tik_instance.if_scope(burst > 0):
                        self.tik_instance.data_move(dst_ub[offset], dst_ub, 0, M, burst, src_stride,
                                                    dst_stride)
                    self.tik_instance.tik_break()
                times.set_as(times * 2)
        return dst_ub

    def up_align_to_K0(self, n, dtype=None):
        if dtype is None:
            dtype = self.dtype

        K0 = 32 // DTYPE_SIZE[dtype]
        return (n + K0 - 1) // K0 * K0

    def calc_vec_rec(self, vec_ub, vec_len):
        """calc rec of the given vec
        :param vec_ub: input tensor in UB
        :param vec_len: input scalar
        :return: rowsum_ub
        """
        dtype = vec_ub.dtype
        vec_len_aligned = self.up_align_to_K0(vec_len)
        vec_rec_ub = self.tik_instance.Tensor(dtype, (vec_len_aligned,), scope=UB, name="li_new_rec_ub")
        mask_len = 256 // DTYPE_SIZE[dtype]
        block_len = 32 // DTYPE_SIZE[dtype]
        work_size = 8 // DTYPE_SIZE[dtype]

        with self.tik_instance.new_stmt_scope(disable_sync=False):
            repeat_times = vec_len // mask_len
            if repeat_times > 0:
                dst_rep_stride = 8
                src_rep_stride = 8

                src_extent_size = (repeat_times - 1) * src_rep_stride * block_len + mask_len
                wk_size_unit = ((src_extent_size + block_len - 1) // block_len) * block_len
                wk_size = work_size * wk_size_unit
                # 定义work_tensor
                work_tensor_ub = self.tik_instance.Tensor(
                    "float32", (wk_size,), name="work_tensor_ub", scope=UB
                )
                # 如果work_tensor有索引，需要写成 work_tensor[index:]
                self.tik_instance.vec_rec_high_preci(
                    mask_len,
                    vec_rec_ub[0:],
                    vec_ub[0:],
                    work_tensor_ub[0:],
                    repeat_times,
                    dst_rep_stride,
                    src_rep_stride,
                )

            mask_len = vec_len - repeat_times * mask_len
            if mask_len > 0:
                wk_size = work_size * ((mask_len + block_len - 1) // block_len) * block_len
                work_tensor_ub2 = self.tik_instance.Tensor(
                    "float32", (wk_size,), name="work_tensor_ub2", scope=UB
                )
                self.tik_instance.vec_rec_high_preci(
                    mask_len,
                    vec_rec_ub[repeat_times * 128:],
                    vec_ub[repeat_times * 128:],
                    work_tensor_ub2[0:],
                    1,
                    0,
                    0,
                )
        return vec_rec_ub

    def row_sum_cube_impl(self, matrix_l1_K1MK0_ed, right_all_ones_matrix_l1, rowsum_ub, m, k):
        """用cube实现矩阵行和：右乘一个shape=(n,1)全一矩阵
        :param matrix_l1_K1MK0_ed: input tensor with shape (K1, M, K0)
        :param right_all_ones_matrix_l1: input tensor with shape (K1*K0, 16)
        :param rowsum_ub: output tensor stores the row sum of input tensor.
        :param m: actual tensor height
        :param k: actual tensor width
        :return: rowsum_ub
        """
        K1, M, K0 = matrix_l1_K1MK0_ed.shape
        K = K1 * K0

        # 调用matmul实现rowsum，结果shape=(m, 16)，取每行的第一个数
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            row_sum_ub_N1MN0 = self.matmul_compute(matrix_l1_K1MK0_ed, right_all_ones_matrix_l1, m, k, 16,
                                                   N1MN0_to_MN=False)
            row_sum_ub_MN_ed = row_sum_ub_N1MN0.reshape((M, 16))

            # row_sum_ub_MN_ed 先转置，然后取一行, 替换原来按行操作: lij_ub[i].set_as(row_sum_ub_MN_ed[i, 0])
            row_sum_ub_trans = self.tik_instance.Tensor(FP16, (16, M), name="row_sum_ub_trans", scope=UB)
            row_sum_ub_trans = self.transpose_matrix(row_sum_ub_MN_ed, row_sum_ub_trans, M, True)
            self.cont_data_mv_1_bust(dst=rowsum_ub, src=row_sum_ub_trans, burst=M // 16)

        return rowsum_ub

    def matmul_compute(self, A_l1, B_l1, m, k, n, N1MN0_to_MN=True, workspace=None):
        """calc matrix multiplication A_l1 * B_l1, and move the result to C_ub,
        then rearrange C_ub
        :param A_l1: input tensor in L1 with shape of (K1, M, K0)
        :param B_l1: input tensor in L1 with shape of (K1, N, K0)
        :param m: the actual number of rows of A_l1
        :param k: the actual number of cols of A_l1
        :param n: the actual number of cols of B_l1
        :param N1MN0_to_MN: Whether reorder the result tensor.
        :return: C_ub with tensor with shape of (M, N) if N1MN0_to_MN else (N1, M, N0)
        """
        M = self.up_align_to_K0(m)
        N = self.up_align_to_K0(n)
        # 根据精度类型，决定matmul返回的ub精度类型，如果fp16, tensor_move做随路精度转换
        if workspace is None:
            C_ub = self.tik_instance.Tensor(FP16, (N // 16, M, 16), name="C_ub", scope=UB)
        else:
            C_ub = workspace
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            # matmul
            C_l0c = self.tik_instance.Tensor(
                FP32, (N // 16, M, 16), scope=L0C, name="C_l0c"
            )  # n1mn0 (n0=16)
            self.tik_instance.matmul(C_l0c, A_l1, B_l1, m, k, n)
            # L0C -> ub, fp32 -> fp16 (tensor_mov可做随路转换)
            self.tik_instance.tensor_mov(C_ub, C_l0c, "m", 1, M * N * DTYPE_SIZE[FP32] // 1024, 0, 0)
        if N1MN0_to_MN:
            return self.N1MN0_TO_MN(C_ub)
        else:
            return C_ub

    def move_vector_from_gm_to_ub(self, dst_tensor, src_tensor, gm_offset, vec_len):
        """load the vector from gm to ub
        :param dst_tensor:
        :param src_tensor:
        :param gm_offset:
        :param vec_len:
        :return:
        """
        a_burst_num = 32 // DTYPE_SIZE[src_tensor.dtype]
        full_tik_blk_num, tail_num = divmod(vec_len, a_burst_num)
        with self.tik_instance.if_scope(full_tik_blk_num > 0):
            self.cont_data_mv_1_bust(dst=dst_tensor, src=src_tensor[gm_offset],
                                     burst=full_tik_blk_num)
        # 地址回退处理尾部数据
        with self.tik_instance.if_scope(tail_num > 0):
            offset = vec_len - a_burst_num
            last_blk_ub = self.tik_instance.Tensor(FP16, (a_burst_num,), name="last_blk_ub", scope=UB)
            self.cont_data_mv_1_bust(dst=last_blk_ub, src=src_tensor[gm_offset + offset], burst=1)
            with self.tik_instance.for_range(0, a_burst_num) as idx:  # offset非32bytes对齐，无法用datamove
                dst_tensor[offset + idx].set_as(last_blk_ub[idx])

    def move_vector_from_ub_to_gm(self, dst_tensor, src_tensor, gm_offset, vec_len):
        """write the vector back to gm
        :param dst_tensor:
        :param src_tensor:
        :param gm_offset:
        :param vec_len:
        :return:
        """
        a_burst_num = 32 // DTYPE_SIZE[src_tensor.dtype]
        full_tik_blk_num = vec_len // a_burst_num
        with self.tik_instance.if_scope(full_tik_blk_num > 0):
            self.cont_data_mv_1_bust(dst=dst_tensor[gm_offset], src=src_tensor,
                                     burst=full_tik_blk_num)
        tail_num = vec_len % a_burst_num
        with self.tik_instance.if_scope(tail_num > 0):
            offset = vec_len - a_burst_num
            tmp_ub = self.tik_instance.Tensor(FP16, (a_burst_num,), name="tmp_ub", scope=UB)
            with self.tik_instance.for_range(0, a_burst_num) as idx:
                tmp_ub[idx].set_as(src_tensor[offset + idx])
            self.cont_data_mv_1_bust(dst=dst_tensor[gm_offset + offset], src=tmp_ub, burst=1)

    def vec_duplicate(self, dst, scalar, size):
        """fill a vector with the given scalar
        :param dst: result tensor in UB
        :param scalar: source operand
        :param size: the amount of element will be processed
        :return:
        """
        repeat_times, remain_size = divmod(size, self.repeat_once_size)
        if repeat_times > 0:
            if repeat_times > self.max_repeat_times:
                max_repeat_times_loops, remain_repeat_times = divmod(repeat_times, self.max_repeat_times)
                for idx in range(max_repeat_times_loops):
                    offset = idx * self.max_repeat_times * self.repeat_once_size
                    self.tik_instance.vec_dup(self.repeat_once_size, dst[offset], scalar, self.max_repeat_times, 8)
                if remain_repeat_times > 0:
                    offset = max_repeat_times_loops * self.max_repeat_times * self.repeat_once_size
                    self.tik_instance.vec_dup(self.repeat_once_size, dst[offset], scalar, remain_repeat_times, 8)
            else:
                self.tik_instance.vec_dup(self.repeat_once_size, dst, scalar, repeat_times, 8)
        if remain_size > 0:
            offset = repeat_times * self.repeat_once_size
            self.tik_instance.vec_dup(remain_size, dst[offset], scalar, 1, 0)

    def vec_and_vec_ele_wise(self, op_func, dst, src0, src1, size):
        """vector and vector element wise
        :param op_func: tik op interface
        :param dst: result tensor in UB
        :param src0: source operand0 in UB
        :param src1: source operand1 in UB
        :param size: the amount of element will be processed
        :return:
        """
        repeat_times, remain_size = divmod(size, self.repeat_once_size)
        if repeat_times > 0:
            if repeat_times > self.max_repeat_times:
                max_repeat_times_loops, remain_repeat_times = divmod(repeat_times, self.max_repeat_times)
                for idx in range(max_repeat_times_loops):
                    offset = idx * self.max_repeat_times * self.repeat_once_size
                    op_func(self.repeat_once_size,
                            dst[offset], src0[offset], src1[offset], self.max_repeat_times, 8, 8, 8)
                if remain_repeat_times > 0:
                    offset = max_repeat_times_loops * self.max_repeat_times * self.repeat_once_size
                    op_func(self.repeat_once_size,
                            dst[offset], src0[offset], src1[offset], remain_repeat_times, 8, 8, 8)
            else:
                op_func(self.repeat_once_size, dst, src0, src1, repeat_times, 8, 8, 8)
        if remain_size > 0:
            offset = repeat_times * self.repeat_once_size
            op_func(remain_size, dst[offset], src0[offset], src1[offset], 1, 0, 0, 0)

    def vec_ele_wise(self, op_func, dst, src, size):
        """vector element wise
        :param op_func: tik op interface
        :param dst: result tensor in UB
        :param src: source operand in UB
        :param size: the amount of element will be processed
        :return:
        """
        repeat_times, remain_size = divmod(size, self.repeat_once_size)
        if repeat_times > 0:
            if repeat_times > self.max_repeat_times:
                max_repeat_times_loops, remain_repeat_times = divmod(repeat_times, self.max_repeat_times)
                for idx in range(max_repeat_times_loops):
                    offset = idx * self.max_repeat_times * self.repeat_once_size
                    op_func(self.repeat_once_size, dst[offset], src[offset], self.max_repeat_times, 8, 8)
                if remain_repeat_times > 0:
                    offset = max_repeat_times_loops * self.max_repeat_times * self.repeat_once_size
                    op_func(self.repeat_once_size, dst[offset], src[offset], remain_repeat_times, 8, 8)
            else:
                op_func(self.repeat_once_size, dst, src, repeat_times, 8, 8)
        if remain_size > 0:
            offset = repeat_times * self.repeat_once_size
            op_func(remain_size, dst[offset], src[offset], 1, 0, 0)

    def vec_and_scalar_ele_wise(self, op_func, dst, src, scalar, size):
        """vector and scalar element wise
        :param op_func: tik op interface
        :param dst: result tensor in UB
        :param src: source operand1 in UB
        :param scalar: source operand2
        :param size: the amount of element will be processed
        :return:
        """
        repeat_times, remain_size = divmod(size, self.repeat_once_size)
        if repeat_times > 0:
            if repeat_times > self.max_repeat_times:
                max_repeat_times_loops, remain_repeat_times = divmod(repeat_times, self.max_repeat_times)
                for idx in range(max_repeat_times_loops):
                    offset = idx * self.max_repeat_times * self.repeat_once_size
                    op_func(self.repeat_once_size, dst[offset], src[offset], scalar, self.max_repeat_times, 8, 8)
                if remain_repeat_times > 0:
                    offset = max_repeat_times_loops * self.max_repeat_times * self.repeat_once_size
                    op_func(self.repeat_once_size, dst[offset], src[offset], scalar, remain_repeat_times, 8, 8)
            else:
                op_func(self.repeat_once_size, dst, src, scalar, repeat_times, 8, 8)
        if remain_size > 0:
            offset = repeat_times * self.repeat_once_size
            op_func(remain_size, dst[offset], src[offset], scalar, 1, 0, 0)

    def vec_reduce(self, op_func, dst, src, size):
        """calc reduce sum, max or min of vector
        :param op_func: tik op interface
        :param dst: result tensor in UB
        :param src: source operand in UB
        :param size: the amount of element will be processed
        :return:
        """
        repeat_times, remain_size = divmod(size, self.repeat_once_size)
        if repeat_times > 0:
            if repeat_times > self.max_repeat_times:
                max_repeat_times_loops, remain_repeat_times = divmod(repeat_times, self.max_repeat_times)
                for idx in range(max_repeat_times_loops):
                    src_offset = idx * self.max_repeat_times * self.repeat_once_size
                    dst_offset = idx * self.max_repeat_times * self.block_size
                    op_func(self.repeat_once_size, dst[dst_offset], src[src_offset], repeat_times, 1, 1, 8)
                if remain_repeat_times > 0:
                    src_offset = max_repeat_times_loops * self.max_repeat_times * self.repeat_once_size
                    dst_offset = max_repeat_times_loops * self.max_repeat_times * self.block_size
                    op_func(self.repeat_once_size, dst[dst_offset], src[src_offset], repeat_times, 1, 1, 8)
            else:
                op_func(self.repeat_once_size, dst, src, repeat_times, 1, 1, 8)
        if remain_size > 0:
            src_offset = repeat_times * self.repeat_once_size
            dst_offset = repeat_times * 16
            op_func(remain_size, dst[dst_offset], src[src_offset], 1, 1, 0)