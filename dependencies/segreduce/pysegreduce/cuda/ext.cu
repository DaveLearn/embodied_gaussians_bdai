#include <cstdint>
#include <pybind11/pybind11.h>
namespace py = pybind11;
// #include <moderngpu/kernel_segreduce.hxx>
// using namespace mgpu;


//#include <nanobind/nanobind.h>
//#include <nanobind/stl/tuple.h>
//namespace nb = nanobind;
//using namespace nb::literals;

// cub reduce
#include <cub/device/device_segmented_reduce.cuh>



__global__ void shortKernel(float * out_d, float * in_d){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<3) out_d[idx]=1.0f;
}

struct mat33 {
    float data[9] = {
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f
    };
};



struct mat33_plus {
    __device__ __forceinline__ mat33 operator()(const mat33& a, const mat33& b) const {
        mat33 res;
        #pragma unroll
        for (int i = 0; i < 9; i++) {
            res.data[i] = a.data[i] + b.data[i];
        }
        return res;
    }
};

struct vec3 {
    float data[3];
};

struct vec3_plus {
    __device__ __forceinline__ vec3 operator()(const vec3& a, const vec3& b) const {
        return {a.data[0] + b.data[0], a.data[1] + b.data[1], a.data[2] + b.data[2]};
    }
};

void reduce_mat33_impl(
    uint64_t d_in_ptr,
    uint64_t start_offsets_in_ptr,
    uint64_t end_offsets_in_ptr,
    const int num_segments,
    uint64_t d_out_ptr,
    uint64_t stream_id
    ) {
    cudaStream_t stream = (cudaStream_t)stream_id;
    void *storage_ptr = nullptr;
    size_t storage_bytes = 0;
    const mat33 zero = {
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f
    };
    cub::DeviceSegmentedReduce::Reduce(
            (void*)storage_ptr,
            storage_bytes,
            (mat33*)d_in_ptr, 
            (mat33*)d_out_ptr, 
            num_segments,
            (int*)start_offsets_in_ptr,
            (int*)end_offsets_in_ptr,
            mat33_plus(),
            zero,
            stream
            );

    cudaMallocAsync(&storage_ptr, storage_bytes, stream);
    cub::DeviceSegmentedReduce::Reduce(
            (void*)storage_ptr,
            storage_bytes,
            (mat33*)d_in_ptr, 
            (mat33*)d_out_ptr, 
            num_segments,
            (int*)start_offsets_in_ptr,
            (int*)end_offsets_in_ptr,
            mat33_plus(),
            zero,
            stream
            );
};


void reduce_vec3_impl(
    uint64_t d_in_ptr,
    uint64_t start_offsets_in_ptr,
    uint64_t end_offsets_in_ptr,
    const int num_segments,
    uint64_t d_out_ptr,
    uint64_t stream_id
    ) {
    cudaStream_t stream = (cudaStream_t)stream_id;
    void *storage_ptr = nullptr;
    size_t storage_bytes = 0;
    cub::DeviceSegmentedReduce::Reduce(
            (void*)storage_ptr,
            storage_bytes,
            (vec3*)d_in_ptr, 
            (vec3*)d_out_ptr, 
            num_segments,
            (int*)start_offsets_in_ptr,
            (int*)end_offsets_in_ptr,
            vec3_plus(),
            vec3{0.0f, 0.0f, 0.0f},
            stream
            );
    cudaMallocAsync(&storage_ptr, storage_bytes, stream);
    cub::DeviceSegmentedReduce::Reduce(
            (void*)storage_ptr,
            storage_bytes,
            (vec3*)d_in_ptr, 
            (vec3*)d_out_ptr, 
            num_segments,
            (int*)start_offsets_in_ptr,
            (int*)end_offsets_in_ptr,
            vec3_plus(),
            vec3{0.0f, 0.0f, 0.0f},
            stream
            );
};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    //m.def("reduce_float", &reduce_impl<float>, "reduce float", py::call_guard<py::gil_scoped_release>());
    m.def("reduce_vec3f", &reduce_vec3_impl, py::call_guard<py::gil_scoped_release>());
    m.def("reduce_mat33f", &reduce_mat33_impl, py::call_guard<py::gil_scoped_release>());
}


