/*!
  | TODO: this file should be in ATen/cuda,
  | not top level
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/CUDAGeneratorImpl.h]

/**
 | Note [CUDA Graph-safe RNG states]
 | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 |
 | Strategy:
 | ~~~~~~~~~
 | (It helps to look at
 | cuda/detail/PhiloxCudaStateRaw.cuh and
 | cuda/detail/UnpackRaw.cuh
 | while you read this.)
 |
 | A CUDA graph containing multiple RNG ops behaves like a
 | single giant kernel from the perspective of ops external
 | to the graph.  During graph capture, logic below records
 | the total of all offset increments that occur in the graphed
 | region, and records the final total as the offset for the
 | entire graph.
 |
 | When the graph reruns, the logic that reruns it
 | increments this device's CUDA generator's offset
 | by that total.
 |
 | Meanwhile, within the graph, at capture time, instead of
 | populating PhiloxCudaStates with the u64 offset pulled
 | directly from the global state, PhiloxCudaState instead
 | holds a pointer to one-element stream-local i64 device tensor
 | holding an initial offset value, and a u64 holding an
 | intra-graph offset. (The intra-graph offset starts from zero
 | when capture begins.)  In each consumer kernel,
 | at::cuda::philox::unpack computes the offset to use for this kernel
 | as intra-graph offset + *initial offset.
 |
 | When the graph reruns, the logic that reruns it first
 | fill_s the initial offset tensor with this device's
 | CUDA generator's current offset.
 |
 | The control flow above ensures graphed execution is bitwise
 | identical to eager execution as long as RNG ops are enqueued
 | from a single thread, even if RNG ops and graphs containing
 | RNG ops are enqueued and run simultaneously on multiple streams.
 |
 | Usage:
 | ~~~~~~
 | PhiloxCudaState in this file, and unpack() in
 | cuda/CUDAGraphsUtils.cuh allow non-divergent use of
 | CUDAGeneratorImpl whether graph capture is underway or not.
 |
 | Each PhiloxCudaState instance should be used for one and only one
 | consumer kernel.
 |
 | Example (see e.g. native/cuda/Dropout.cu):
 |
 | #include <ATen/CUDAGeneratorImpl.h>
 | #include <ATen/cuda/CUDAGraphsUtils.cuh>
 |
 | __global__ void kernel(..., PhiloxCudaState philox_args) {
 |   auto seeds = at::cuda::philox::unpack(philox_args);
 |   IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
 |   curandStatePhilox4_32_10_t state;
 |   curand_init(std::get<0>(seeds), // seed
 |               idx,                // per-thread subsequence
 |               std::get<1>(seeds), // offset in subsequence
 |               &state);
 |   ...
 | }
 |
 | host_caller(...) {
 |   PhiloxCudaState rng_engine_inputs;
 |   {
 |     // See Note [Acquire lock when using random generators]
 |     std::lock_guard<std::mutex> lock(gen->mutex_);
 |
 |     // gen could be HostState or DevState here! No divergent code needed!
 |     rng_engine_inputs = gen->philox_cuda_state(offset_increment);
 |   }
 |   kernel<<<...>>>(..., rng_engine_inputs);
 | }
 |
 */
pub struct CUDAGeneratorImpl {
    base:                     GeneratorImpl,
    seed:                     u64, // default = default_rng_seed_val
    philox_offset_per_thread: u64, // default = 0
    offset_extragraph:        *mut i64,
    offset_intragraph:        u32, // default = 0
    graph_expects_this_gen:   bool, // default = false
}

impl CUDAGeneratorImpl {

    pub fn new(device_index: DeviceIndex) -> Self {

        let device_index: DeviceIndex = device_index.unwrap_or(-1);

        todo!();

        /*
        
        */
    }

    /* ----------- CUDAGeneratorImpl methods  ----------- */

    pub fn clone(&self) -> Arc<CUDAGeneratorImpl> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_current_seed(&mut self, seed: u64)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn current_seed(&self) -> u64 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn seed(&mut self) -> u64 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_state(&mut self, new_state: &TensorImpl)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_state(&self) -> IntrusivePtr<TensorImpl> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_philox_offset_per_thread(&mut self, offset: u64)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn philox_offset_per_thread(&self) -> u64 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn capture_prologue(&mut self, offset_extragraph: *mut i64)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn capture_epilogue(&mut self) -> u64 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn philox_cuda_state(&mut self, increment: u64) -> PhiloxCudaState {
        
        todo!();
        /*
        
        */
    }

    /**
      | Temporarily accommodates call sites
      | that use philox_engine_inputs.
      | 
      | Allows incremental refactor of call
      | sites to use philox_cuda_state.
      |
      */
    pub fn philox_engine_inputs(&mut self, increment: u64) -> (u64,u64) {
        
        todo!();
        /*
        
        */
    }
    
    pub fn device_type() -> DeviceType {
        
        todo!();
        /*
        
        */
    }
    
    pub fn clone_impl(&self) -> *mut CUDAGeneratorImpl {
        
        todo!();
        /*
        
        */
    }
}

pub fn get_default_cuda_generator(device_index: DeviceIndex) -> &Generator {

    let device_index: DeviceIndex = device_index.unwrap_or(-1);

    todo!();
        /*
        
        */
}

pub fn create_cuda_generator(device_index: DeviceIndex) -> Generator {

    let device_index: DeviceIndex = device_index.unwrap_or(-1);

    todo!();
        /*
        
        */
}
