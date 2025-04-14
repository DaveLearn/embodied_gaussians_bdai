import torch
import pysegreduce
import warp as wp

wp.config.mode = "debug"
# wp.config.verify_cuda = True
wp.init()


@wp.kernel
def simple_add_kernel(input: wp.array(dtype=wp.float32), 
                      add: wp.float32,
                      output: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    output[tid] = input[tid] + add


# xyz = torch.empty(0, 3, dtype=torch.float32).cuda()
# indices = torch.empty(0, 1, dtype=torch.int32).cuda()
# body_ids = torch.empty(0, 1, dtype=torch.int32).cuda()
# out = torch.empty(0, 3, dtype=torch.float32).cuda()

# def add_body(num_points: int, body_id: int):
#     global xyz, indices, body_ids, out
#     prev_num_points = xyz.shape[0]
#     xyz = torch.cat((xyz, torch.randn(num_points, 3).float().cuda()), 0)
#     indices = torch.cat((indices, torch.arange(prev_num_points, prev_num_points + num_points).reshape(-1, 1).int().cuda()), 0)
#     body_ids = torch.cat((body_ids, torch.tensor([body_id]).repeat(num_points).reshape(-1, 1).cuda().int()), 0)
#     out = torch.cat((out, torch.zeros(1, 3).float().cuda()), 0)

# add_body(2, 0)
# add_body(3, 1)
# # print(xyz)

# # print(out)

# xyz_warp = wp.from_torch(xyz, dtype=wp.vec3f)
# indices_warp = wp.from_torch(indices, dtype=wp.int32)
# body_ids_warp = wp.from_torch(body_ids, dtype=wp.int32)
# out = torch.zeros_like(xyz)
# out_warp = wp.from_torch(out, dtype=wp.vec3f)


# xyz_cp = cp.asarray(xyz)
# out_cp = cp.asarray(out)

# print(indices_warp.size)
# num_bytes = pysegreduce.prepare_reduce(
#     xyz_warp.ptr,
#     5,
#     out_warp.ptr)

# storage = wp.empty(num_bytes, dtype=wp.uint8)

# pysegreduce.reduce(storage.ptr,
#         num_bytes,
#         xyz_warp.ptr,
#         5,
#         centroids_warp.ptr
#         )


def test_module_works():
    num_values = 10
    start_idx = torch.tensor([0, 3]).int().cuda()
    end_idx = torch.tensor([3, 10]).int().cuda()
    start_idx_warp = wp.from_torch(start_idx, dtype=wp.int32)
    end_idx_warp = wp.from_torch(end_idx, dtype=wp.int32)

    stream = wp.Stream()
    data_in = wp.ones(num_values, dtype=wp.mat33f, device="cuda")
    num_segments = start_idx.shape[0]
    data_out = wp.ones(num_segments, dtype=wp.mat33f, device="cuda")
    pysegreduce.reduce_mat33f(
            data_in.ptr, 
            start_idx_warp.ptr,
            end_idx_warp.ptr,
            num_segments,
            data_out.ptr, 
            0)

    assert torch.all(torch.tensor(data_out)[0] == torch.ones((3,3), device="cuda") * 3), f"output: {wp.to_torch(data_out)} != {torch.ones((3,3)) * 3}"
    assert torch.all(torch.tensor(data_out)[1] == torch.ones((3,3), device="cuda") * 7), f"output: {wp.to_torch(data_out)} != {torch.ones((3,3)) * 7}"
    










