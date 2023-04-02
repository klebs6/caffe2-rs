crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp]

lazy_static!{
    /*
    // Ensures we only call cudaGetDeviceCount only once.
    static std::once_flag num_gpu_init_flag;

    // Total number of gpus in the system.
    static i64 num_gpus;

    // Ensures default_gens_cuda is initialized once.
    static std::deque<std::once_flag> cuda_gens_init_flag;

    // Default, global CUDA generators, one per GPU.
    static std::vector<Generator> default_gens_cuda;
    */
}

/**
  | Populates the global variables related
  | to CUDA generators
  | 
  | Warning: this function must only be
  | called once!
  |
  */
pub fn init_cuda_gen_vector()  {
    
    todo!();
        /*
            num_gpus = c10::cuda::device_count();
      cuda_gens_init_flag.resize(num_gpus);
      default_gens_cuda.resize(num_gpus);
        */
}

/**
  | PyTorch maintains a collection of default
  | generators that get initialized once.
  | 
  | The purpose of these default generators
  | is to maintain a global running state
  | of the pseudo random number generation,
  | when a user does not explicitly mention
  | any generator. getDefaultCUDAGenerator
  | gets the default generator for a particular
  | cuda device.
  |
  */
pub fn get_default_cuda_generator(device_index: DeviceIndex) -> &Generator {
    
    todo!();
        /*
            std::call_once(num_gpu_init_flag, initCUDAGenVector);
      DeviceIndex idx = device_index;
      if (idx == -1) {
        idx = c10::cuda::current_device();
      } else {
        TORCH_CHECK(idx >= 0 && idx < num_gpus);
      }
      std::call_once(cuda_gens_init_flag[idx], [&] {
        default_gens_cuda[idx] = make_generator<CUDAGeneratorImpl>(idx);
        default_gens_cuda[idx].seed();
      });
      return default_gens_cuda[idx];
        */
}

/**
  | Utility to create a CUDAGeneratorImpl.
  | Returns a shared_ptr
  |
  */
pub fn create_cuda_generator(device_index: DeviceIndex) -> Generator {
    
    todo!();
        /*
            std::call_once(num_gpu_init_flag, initCUDAGenVector);
      DeviceIndex idx = device_index;
      if (idx == -1) {
        idx = c10::cuda::current_device();
      }
      TORCH_CHECK(idx >= 0 && idx < num_gpus, "The device_index is invalid.");
      auto gen = make_generator<CUDAGeneratorImpl>(idx);
      auto cuda_gen = check_generator<CUDAGeneratorImpl>(gen);
      cuda_gen->set_current_seed(default_rng_seed_val);
      cuda_gen->set_philox_offset_per_thread(0);
      return gen;
        */
}

/**
  | Note [Why enforce RNG offset % 4 == 0?]
  | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  | 
  | Curand philox does allow offsets that
  | aren't a multiple of 4.
  | 
  | But jit kernels don't use curand, they
  | use a custom "Philox" class (see torch/csrc/jit/tensorexpr/cuda_random.h
  | or torch/csrc/jit/codegen/cuda/runtime/random_numbers.cu).
  | 
  | The "Philox" constructor computes
  | offset/4 (a u64 division) to locate
  | its internal start in its virtual bitstream
  | viewed as 128-bit chunks, then, when
  | called in a thread, returns one 32-bit
  | chunk at a time from that start in the
  | bitstream.
  | 
  | In other words, if the incoming offset
  | is not a multiple of 4, each thread might
  | repeat some previously-generated
  | 32-bit values in the bitstream. See
  | https://github.com/pytorch/pytorch/pull/50169.
  |
  */
pub const CAPTURE_DEFAULT_GENS_MSG: &'static str =
"In regions captured by CUDA graphs, you may only use the default CUDA RNG \
generator on the device that's current when capture begins. \
If you need a non-default (user-supplied) generator, or a generator on another \
device, please file an issue.";

impl CUDAGeneratorImpl {
    
    /**
      | CUDAGeneratorImpl class implementation
      |
      */
    pub fn new(device_index: DeviceIndex) -> Self {
    
        todo!();
        /*


            : c10::GeneratorImpl{Device(DeviceType::CUDA, device_index),
                  DispatchKeySet(c10::DispatchKey::CUDA)} 

      at::cuda::assertNotCapturing("Cannot construct a new CUDAGeneratorImpl");
        */
    }
    
    /**
      | Sets the seed to be used by curandStatePhilox4_32_10
      | 
      | Resets the philox_offset_per_thread_
      | to 0
      | 
      | See Note [Acquire lock when using random
      | generators]
      |
      */
    pub fn set_current_seed(&mut self, seed: u64)  {
        
        todo!();
        /*
            at::cuda::assertNotCapturing("Cannot call CUDAGeneratorImpl::set_current_seed");
      seed_ = seed;
      philox_offset_per_thread_ = 0;
        */
    }

    /**
      | Gets the current seed of CUDAGeneratorImpl.
      |
      */
    pub fn current_seed(&self) -> u64 {
        
        todo!();
        /*
            // Debatable if current_seed() should be allowed in captured regions.
      // Conservatively disallow it for now.
      at::cuda::assertNotCapturing("Cannot call CUDAGeneratorImpl::current_seed");
      return seed_;
        */
    }

    /**
      | Gets a nondeterministic random number
      | from /dev/urandom or time, seeds the
      | CPUGeneratorImpl with it and then returns
      | that number.
      | 
      | FIXME: You can move this function to
      | Generator.cpp if the algorithm in getNonDeterministicRandom
      | is unified for both CPU and CUDA
      |
      */
    pub fn seed(&mut self) -> u64 {
        
        todo!();
        /*
            at::cuda::assertNotCapturing("Cannot call CUDAGeneratorImpl::seed");
      auto random = c10::detail::getNonDeterministicRandom(true);
      this->set_current_seed(random);
      return random;
        */
    }

    /**
      | Gets the current internal state of CUDAGeneratorImpl.
      | The internal state is returned as a CPU
      | byte tensor.
      |
      */
    pub fn get_state(&self) -> IntrusivePtr<TensorImpl> {
        
        todo!();
        /*
            // The RNG state comprises the seed, and an offset used for Philox.
      // The following line is just here for BC reason. sizeof curandStateMtgp32 is 4120.
      // It used to be static const usize states_size = MAX_NUM_BLOCKS * sizeof(curandStateMtgp32);
      // MAX_NUM_BLOCKS was 200 and sizeof(curandStateMtgp32) is 4120. Hardcoding these numbers here
      // because this is just host side code and we don't want to worry about linking with cuda
      static const usize states_size = 200 * sizeof(4120);
      static const usize seed_size = sizeof(u64);
      static const usize offset_size = sizeof(i64);
      static const usize total_size = states_size + seed_size + offset_size;

      auto state_tensor = detail::empty_cpu({(i64)total_size}, ScalarType::Byte, nullopt, nullopt, nullopt, nullopt);
      auto rng_state = state_tensor.data_ptr<u8>();
      // since curandStateMTGP is not used anymore, fill gen_states of THCGenerator with deterministic garbage value of -1
      // gen_states in THCGenerator struct was an array of curandStateMtgp32s.
      memset(rng_state, -1, states_size);
      auto current_seed = this->current_seed();
      auto offset = static_cast<i64>(this->philox_offset_per_thread()); // Note that old THCGeneratorState had offset as atomic<i64>
      memcpy(rng_state + states_size, &current_seed, seed_size);
      memcpy(rng_state + states_size + seed_size, &offset, offset_size);

      return state_tensor.getIntrusivePtr();
        */
    }

    /**
      | Sets the internal state of CUDAGeneratorImpl.
      | The new internal state must be a strided
      | CPU byte tensor and have appropriate
      | size. See comments of CUDAGeneratorImpl::state
      | for information about the layout and
      | size of the internal state.
      |
      */
    pub fn set_state(&mut self, new_state: &TensorImpl)  {
        
        todo!();
        /*
            static const usize states_size = 200 * sizeof(4120); // this line is just here for BC reason
      static const usize seed_size = sizeof(u64);
      static const usize offset_size = sizeof(i64);
      static const usize total_size = states_size + seed_size + offset_size;

      detail::check_rng_state(new_state);

      bool no_philox_seed = false;
      auto new_state_size = new_state.numel();
      if (new_state_size == total_size - offset_size) {
        no_philox_seed = true;
      } else {
        TORCH_CHECK(new_state_size == total_size, "RNG state is wrong size");
      }

      u64 input_seed;
      auto new_rng_state = new_state.data<u8>();
      memcpy(&input_seed, new_rng_state + states_size, seed_size);
      this->set_current_seed(input_seed);
      i64 philox_offset = 0;
      if (!no_philox_seed) {
        memcpy(&philox_offset, new_rng_state + states_size + seed_size, offset_size);
      }
      this->set_philox_offset_per_thread(static_cast<u64>(philox_offset));
        */
    }

    /**
      | Sets the philox_offset_per_thread_
      | to be used by curandStatePhilox4_32_10
      | 
      | See Note [Acquire lock when using random
      | generators]
      |
      */
    pub fn set_philox_offset_per_thread(&mut self, offset: u64)  {
        
        todo!();
        /*
            cuda::assertNotCapturing("Cannot call CUDAGeneratorImpl::set_philox_offset_per_thread");
      // see Note [Why enforce RNG offset % 4 == 0?]
      TORCH_CHECK(offset % 4 == 0, "offset must be a multiple of 4");
      philox_offset_per_thread_ = offset;
        */
    }

    /**
      | Gets the current philox_offset_per_thread_
      | of CUDAGeneratorImpl.
      |
      */
    pub fn philox_offset_per_thread(&self) -> u64 {
        
        todo!();
        /*
            cuda::assertNotCapturing("Cannot call CUDAGeneratorImpl::philox_offset_per_thread");
      return philox_offset_per_thread_;
        */
    }

    /**
      | Called by CUDAGraph to prepare this
      | instance for a graph capture region.
      | offset_extragraph is the initial offset
      | at the start of the graphed region. offset_intragraph
      | tracks the offset in the graphed region.
      |
      */
    pub fn capture_prologue(&mut self, offset_extragraph: *mut i64)  {
        
        todo!();
        /*
            offset_extragraph_ = offset_extragraph;
      offset_intragraph_ = 0;
      graph_expects_this_gen_ = true;
        */
    }

    /**
      | Called by CUDAGraph to finalize a graph
      | capture region for this instance.
      |
      */
    pub fn capture_epilogue(&mut self) -> u64 {
        
        todo!();
        /*
            graph_expects_this_gen_ = false;
      return offset_intragraph_;
        */
    }

    /**
      | Gets the seed and philox offset value
      | to be used in curandStatePhilox4_32_10,
      | in an opaque PhiloxCudaState that's
      | safe and can be used non-divergently
      | in callers whether CUDA graph capture
      | is underway or not. See
      | 
      | Note [CUDA Graph-safe RNG states]
      | 
      | Each kernel using philox has to sensibly
      | increment offset for future users of
      | philox. So it gets the "old" value for
      | itself (before add), and tells subsequent
      | users which offset they should use,
      | since only the kernel knows how many
      | randoms it intends to generate.
      | 
      | Increment should be at least the number
      | of curand() random numbers used in each
      | thread. It is the user's responsibility
      | to make sure the increment for philox
      | is never smaller than the number of curand()
      | calls. Increment value > the number
      | of curand() calls won't harm but anything
      | less would mean that you would be reusing
      | random values from previous calls.
      | 
      | See Note [Acquire lock when using random
      | generators]
      |
      */
    pub fn philox_cuda_state(&mut self, increment: u64) -> PhiloxCudaState {
        
        todo!();
        /*
            // rounds increment up to the nearest multiple of 4
      increment = ((increment + 3) / 4) * 4;
      if (cuda::currentStreamCaptureStatus() != cuda::CaptureStatus::None) {
        TORCH_CHECK(graph_expects_this_gen_,
                    "philox_cuda_state for an unexpected CUDA generator used during capture. "
                    CAPTURE_DEFAULT_GENS_MSG);
        // see Note [Why enforce RNG offset % 4 == 0?]
        TORCH_INTERNAL_ASSERT(this->offset_intragraph_ % 4 == 0);
        u32 offset = this->offset_intragraph_;
        TORCH_INTERNAL_ASSERT(this->offset_intragraph_ <=
                              u32::max - increment);
        this->offset_intragraph_ += increment;
        return PhiloxCudaState(this->seed_,
                               this->offset_extragraph_,
                               offset);
      } else {
        TORCH_CHECK(!graph_expects_this_gen_,
                    "CUDA generator expects graph capture to be underway, "
                    "but the current stream is not capturing.");
        // see Note [Why enforce RNG offset % 4 == 0?]
        TORCH_INTERNAL_ASSERT(this->philox_offset_per_thread_ % 4 == 0);
        u64 offset = this->philox_offset_per_thread_;
        this->philox_offset_per_thread_ += increment;
        return PhiloxCudaState(this->seed_, offset);
      }
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
            cuda::assertNotCapturing("Refactor this op to use CUDAGeneratorImpl::philox_cuda_state. "
                                   "Cannot call CUDAGeneratorImpl::philox_engine_inputs");
      // rounds increment up to the nearest multiple of 4
      increment = ((increment + 3) / 4) * 4;
      // see Note [Why enforce RNG offset % 4 == 0?]
      TORCH_INTERNAL_ASSERT(this->philox_offset_per_thread_ % 4 == 0);
      u64 offset = this->philox_offset_per_thread_;
      this->philox_offset_per_thread_ += increment;
      return make_pair(this->seed_, offset);
        */
    }

    /**
      | Gets the DeviceType of CUDAGeneratorImpl.
      | 
      | Used for type checking during run time.
      |
      */
    pub fn device_type(&mut self) -> DeviceType {
        
        todo!();
        /*
            return DeviceType::Cuda;
        */
    }

    /**
      | Public clone method implementation
      | 
      | See Note [Acquire lock when using random
      | generators]
      |
      */
    pub fn clone(&self) -> Arc<CUDAGeneratorImpl> {
        
        todo!();
        /*
            return shared_ptr<CUDAGeneratorImpl>(this->clone_impl());
        */
    }

    /**
      | Private clone method implementation
      | 
      | See Note [Acquire lock when using random
      | generators]
      |
      */
    pub fn clone_impl(&self) -> *mut CUDAGeneratorImpl {
        
        todo!();
        /*
            cuda::assertNotCapturing("Cannot call CUDAGeneratorImpl::clone_impl");
      auto gen = new CUDAGeneratorImpl(this->device().index());
      gen->set_current_seed(this->seed_);
      gen->set_philox_offset_per_thread(this->philox_offset_per_thread_);
      return gen;
        */
    }
}
