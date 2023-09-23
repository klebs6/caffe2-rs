crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cuda/DistributionTemplates.h]

/**
  | launch bounds used for kernels utilizing
  | TensorIterator
  |
  */
pub const BLOCK_SIZE_BOUND: u32 = 256;
pub const GRID_SIZE_BOUND: u32 = 4;

/**
  | number of randoms given by distributions like
  | curand_uniform4, curand_uniform2_double used in
  | calculating philox offset.
  |
  */
pub const CURAND4_ENGINE_CALLS: u32 = 4;

/**
  | utility function that calculates proper
  | philox_offset for distributions utilizing
  | TensorIterator. For distributions using
  | TensorIterator, we are using a grid-stride loop
  | with each thread yielding one element per
  | thread. For the edge of the grid-stride loop,
  | if the tensor size is large, the unroll loop
  | will kick in and the float4 from curand4 will
  | start getting utilized (for common tensor
  | sizes, we end up using rand.x from each
  | thread). Hence, the philox_offset is (number of
  | elements per thread * number of engine calls),
  | which makes sure that philox offset increment
  | is not less than the number of randoms used in
  | each thread.
  |
  */
pub fn calc_execution_policy(total_elements: i64) -> (u64,dim3,dim3) {
    
    todo!();
        /*
            const u64 numel = static_cast<u64>(total_elements);
      const u32 block_size = block_size_bound;
      const u32 unroll = curand4_engine_calls;
      dim3 dim_block(block_size);
      dim3 grid((numel + block_size - 1) / block_size);
      u32 blocks_per_sm = getCurrentDeviceProperties()->maxThreadsPerMultiProcessor / block_size;
      grid.x = min(
          static_cast<u32>(getCurrentDeviceProperties()->multiProcessorCount) * blocks_per_sm,
          grid.x);
      //number of times random will be generated per thread, to offset philox counter in thc random state
      u64 counter_offset = ((numel - 1) / (block_size * grid.x * unroll) + 1)
                                    * curand4_engine_calls;
      return make_tuple(counter_offset, grid, dim_block);
        */
}

/**
  | grid stride loop kernel for distributions
  |
  */
#[__global__]
#[launch_bounds(block_size_bound, grid_size_bound)]
pub fn distribution_elementwise_grid_stride_kernel<accscalar_t, const unroll_factor: i32, dist_t, transform_t>(
        numel:          i32,
        philox_args:    PhiloxCudaState,
        dist_func:      Dist,
        transform_func: Transform)  {

    todo!();
        /*
            auto seeds = philox::unpack(philox_args);
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      curandStatePhilox4_32_10_t state;
      curand_init(get<0>(seeds),
                  idx,
                  get<1>(seeds),
                  &state);

      int rounded_size = ((numel - 1)/(blockDim.x * gridDim.x * unroll_factor)+1) *
          blockDim.x * gridDim.x * unroll_factor;
      for(int linear_index = idx; linear_index < rounded_size; linear_index += blockDim.x * gridDim.x * unroll_factor) {
        auto rand = dist_func(&state);
        #pragma unroll
        for (int ii = 0; ii < unroll_factor; ii++) {
          int li = linear_index + blockDim.x * gridDim.x * ii;
          if (li < numel) {
            transform_func(li, static_cast<accscalar_t>((&rand.x)[ii]));
          }
        }
        __syncthreads();
      }
        */
}

/**
  | distribution_nullary_kernel is analogous
  | to gpu_kernel in
  | 
  | ATen/native/cuda/Loops.cuh. Like
  | gpu_kernel, it uses
  | 
  | TensorIterator to launch a kernel.
  | However, the differences are
  | 
  | - it launches a grid-stride loop based
  | kernel. The kernel is not generic like
  | elementwise_kernel in Loops.cuh and
  | is specialized for the distribution
  | kernels here.
  | 
  | - For big size tensors, we can launch
  | multiple kernels recursively (i.e.
  | if (!iter.can_use_32bit_indexing()))
  | and hence, the philox offset calculation
  | is done in this function.
  | 
  | FIXME: Can we specialize elementwise_kernel
  | and launch_kernel in Loops.cuh to have
  | grid-stride loop kernel and then use
  | that to launch our distribution kernels?
  | Note that we need a grid-stride loop
  | kernel because, we found by testing
  | that it achieves peak effective bandwidth.
  |
  */
pub fn distribution_nullary_kernel<Scalar, accscalar_t, const unroll_factor: i32, RNG, dist_t, transform_t>(
        iter:           &mut TensorIteratorBase,
        gen:            RNG,
        dist_func:      &Dist,
        transform_func: Transform)  {

    todo!();
        /*
            static_assert(unroll_factor >= 1, "unroll_factor must be >= 1.");
      i64 numel = iter.numel();
      if (numel == 0) {
        return;
      }

      auto execution_policy = calc_execution_policy(numel);
      auto counter_offset = get<0>(execution_policy);
      auto grid = get<1>(execution_policy);
      auto block = get<2>(execution_policy);
      PhiloxCudaState rng_engine_inputs;
      {
        // See Note [Acquire lock when using random generators]
        lock_guard<mutex> lock(gen->mutex_);
        rng_engine_inputs = gen->philox_cuda_state(counter_offset);
      }

      if (!iter.can_use_32bit_indexing()) {
        for (auto& sub_iter : iter.with_32bit_indexing()) {
          distribution_nullary_kernel<Scalar, accscalar_t, unroll_factor>(sub_iter,
            gen, dist_func, transform_func);
        }
        return;
      }

      char* out_data = (char*)iter.data_ptr(0);

      auto stream = getCurrentCUDAStream();
      if (iter.is_trivial_1d()) {
        auto strides = iter.get_inner_strides();
        int stride0 = strides[0];
        distribution_elementwise_grid_stride_kernel<accscalar_t, unroll_factor><<<grid, block, 0, stream>>>(
          numel,
          rng_engine_inputs,
          dist_func,
          [=]__device__(int idx, accscalar_t rand) {
            Scalar* out = (Scalar*)&out_data[stride0 * idx];
            *out = transform_func(rand);
          }
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        auto offset_calc = make_offset_calculator<1>(iter);
        distribution_elementwise_grid_stride_kernel<accscalar_t, unroll_factor><<<grid, block, 0, stream>>>(
          numel,
          rng_engine_inputs,
          dist_func,
          [=]__device__(int idx, accscalar_t rand) {
            auto offsets = offset_calc.get(idx);
            Scalar* out = (Scalar*)&out_data[offsets[0]];
            *out = transform_func(rand);
          }
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
        */
}

/// Binary kernel
///
#[__global__]
pub fn distribution_binary_elementwise_kernel<A1, A2, R, F: Fn(A1, A2) -> R, InpOffsetCalc, OutOffsetCalc>(
    numel:        i32,
    f:            F,
    philox_args:  PhiloxCudaState,
    output_data:  *mut R,
    input_data_1: *const A1,
    input_data_2: *const A2,
    inp_calc:     InpOffsetCalc,
    out_calc:     OutOffsetCalc)  {

    todo!();
        /*
            auto seeds = philox::unpack(philox_args);

      using input_t_1 = typename function_traits<func_t>::template arg<1>::type;
      using input_t_2 = typename function_traits<func_t>::template arg<2>::type;

      input_t_1 inputs_1[thread_work_size];
      input_t_2 inputs_2[thread_work_size];

      int base_index = BLOCK_WORK_SIZE * blockIdx.x;
      int remaining = min<int>(numel - base_index, BLOCK_WORK_SIZE);

      curandStatePhilox4_32_10_t state;
      curand_init(get<0>(seeds),
                  blockIdx.x * blockDim.x + threadIdx.x,
                  get<1>(seeds),
                  &state);

      // load data into registers
      int thread_idx = threadIdx.x;
      #pragma unroll
      for (int i = 0; i < thread_work_size; i++) {
        if (thread_idx >= remaining) {
          break;
        }
        int input_idx = thread_idx + base_index;
        auto offsets = inp_calc.get(input_idx);
        inputs_1[i] = input_data_1[offsets[0]];
        inputs_2[i] = input_data_2[offsets[1]];

        thread_idx += num_threads;
      }

      // compute and store
      thread_idx = threadIdx.x;
      #pragma unroll
      for (int i = 0; i < thread_work_size; i++) {
        if (thread_idx >= remaining) {
          break;
        }
        int input_idx = thread_idx + base_index;
        auto offsets = out_calc.get(input_idx);
        output_data[offsets[0]] = f(state, inputs_1[i], inputs_2[i]);
        thread_idx += num_threads;
      }
        */
}

pub fn distribution_binary_kernel<func_t>(
    iter:        &mut TensorIterator,
    philox_args: PhiloxCudaState,
    f:           &Func)  {

    todo!();
        /*
            static_assert(is_same<typename function_traits<func_t>::template arg<0>::type, curandStatePhilox4_32_10_t&>::value, "the first argument of functor must be curandStatePhilox4_32_10_t");
      using input_t_1 = typename function_traits<func_t>::template arg<1>::type;
      using input_t_2 = typename function_traits<func_t>::template arg<2>::type;
      using output_t = typename function_traits<func_t>::result_type;

      if (!iter.can_use_32bit_indexing()) {
        for (auto& sub_iter : iter.with_32bit_indexing()) {
          distribution_binary_kernel(sub_iter, philox_args, f);
        }
        return;
      }

      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(iter.can_use_32bit_indexing());

      i64 numel = iter.numel();
      if (numel == 0) {
        return;
      }

      output_t *output_data = static_cast<output_t *>(iter.data_ptr(0));
      const input_t_1 *input_data_1 = static_cast<const input_t_1 *>(iter.data_ptr(1));
      const input_t_2 *input_data_2 = static_cast<const input_t_2 *>(iter.data_ptr(2));

      i64 grid = (numel + block_work_size - 1) / block_work_size;
      auto stream = getCurrentCUDAStream();

      if (iter.is_contiguous()) {
        distribution_binary_elementwise_kernel<<<grid,num_threads, 0, stream>>>(
            numel, f, philox_args, output_data, input_data_1, input_data_2,
            TrivialOffsetCalculator<2>(), TrivialOffsetCalculator<1>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        distribution_binary_elementwise_kernel<<<grid, num_threads, 0, stream>>>(
            numel, f, philox_args, output_data, input_data_1, input_data_2,
            make_input_offset_calculator<2>(iter), make_output_offset_calculator(iter));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
        */
}

// ==================================================== Random ========================================================

pub fn random_from_to_kernel<RNG>(
        iter:  &mut TensorIteratorBase,
        range: u64,
        base:  i64,
        gen:   RNG)  {

    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND3(ScalarType::Bool, ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "random_from_to_kernel_cuda", [&] {
        if ((
          is_same<Scalar, i64>::value ||
          is_same<Scalar, double>::value ||
          is_same<Scalar, float>::value ||
          is_same<Scalar, BFloat16>::value) && range >= 1ULL << 32)
        {
          // define lambda to mod with range and add base
          auto random_func = [range, base] __device__ (u64 rand) {
            return transformation::uniform_int_from_to<Scalar>(rand, range, base);
          };
          distribution_nullary_kernel<Scalar, u64, curand4_engine_calls/2>(iter,
            gen,
            [] __device__ (curandStatePhilox4_32_10_t* state) -> ulonglong2 {
              ulonglong2 ret;
              uint4 rand_val = curand4(state);
              ret.x = (static_cast<u64>(rand_val.x) << 32) | rand_val.y;
              ret.y = (static_cast<u64>(rand_val.z) << 32) | rand_val.w;
              return ret;
            },
            random_func);
        } else {
          auto random_func = [range, base] __device__ (u32 rand) {
            return transformation::uniform_int_from_to<Scalar>(rand, range, base);
          };
          distribution_nullary_kernel<Scalar, u32, curand4_engine_calls>(iter,
            gen,
            [] __device__ (curandStatePhilox4_32_10_t* state) {
              return curand4(state);
            },
            random_func);
        }
       });
        */
}

/**
  | This is the special kernel to handle single
  | specific case:
  |
  | from(inclusive)
  | = numeric_limits<i64>::lowest()
  |
  | to(exclusive) = None (=
  | i64::max + 1)
  */
pub fn random_full_64_bits_range_kernel<RNG>(
        iter: &mut TensorIteratorBase,
        gen:  RNG)  {

    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND(ScalarType::BFloat16, iter.dtype(), "random_full_64_bits_range_kernel_cuda", [&] {
        if (is_same<Scalar, i64>::value ||
            is_same<Scalar, double>::value ||
            is_same<Scalar, float>::value ||
            is_same<Scalar, BFloat16>::value) {
          auto random_func = [] __device__ (u64 rand) {
            return transformation::uniform_int_full_range<Scalar>(rand);
          };
          distribution_nullary_kernel<Scalar, u64, curand4_engine_calls/2>(iter,
            gen,
            [] __device__ (curandStatePhilox4_32_10_t* state) -> ulonglong2 {
              ulonglong2 ret;
              uint4 rand_val = curand4(state);
              ret.x = (static_cast<u64>(rand_val.x) << 32) | rand_val.y;
              ret.y = (static_cast<u64>(rand_val.z) << 32) | rand_val.w;
              return ret;
            },
            random_func);
        } else {
          TORCH_CHECK(false, "random_full_64_bits_range_kernel_cuda handles only int64, double, float and bfloat16");
        }
      });
        */
}

pub struct RandomFromToKernel<RNG> {

}

impl RandomFromToKernel<RNG> {
    
    pub fn invoke(&mut self, 
        iter:  &mut TensorIteratorBase,
        range: u64,
        base:  i64,
        gen:   Option<dyn GeneratorInterface>)  {
        
        todo!();
        /*
            random_from_to_kernel(iter, range, base, check_generator<RNG>(gen));
        */
    }
    
    pub fn invoke(
        &mut self, 
        iter: &mut TensorIteratorBase,
        gen:  Option<dyn GeneratorInterface>

    ) {
        
        todo!();
        /*
            random_full_64_bits_range_kernel(iter, check_generator<RNG>(gen));
        */
    }
}

pub fn random_kernel<RNG>(
        iter: &mut TensorIteratorBase,
        gen:  RNG)  {

    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Bool, iter.dtype(), "random_kernel_cuda", [&] {
        if (is_same<Scalar, double>::value || is_same<Scalar, i64>::value) {
          auto random_func = [] __device__ (u64 rand) {
            return transformation::uniform_int<Scalar>(rand);
          };
          distribution_nullary_kernel<Scalar, u64, curand4_engine_calls/2>(iter, gen,
            [] __device__ (curandStatePhilox4_32_10_t* state) -> ulonglong2 {
              ulonglong2 ret;
              uint4 rand_val = curand4(state);
              ret.x = (static_cast<u64>(rand_val.x) << 32) | rand_val.y;
              ret.y = (static_cast<u64>(rand_val.z) << 32) | rand_val.w;
              return ret;
            },
            random_func);
        } else {
          auto random_func = [] __device__ (u32 rand) {
            return transformation::uniform_int<Scalar>(rand);
          };
          distribution_nullary_kernel<Scalar, u32, curand4_engine_calls>(iter,
            gen,
            [] __device__ (curandStatePhilox4_32_10_t* state) {
              return curand4(state);
            },
            random_func);
        }
      });
        */
}

pub struct RandomKernel<RNG> {

}

impl RandomKernel<RNG> {
    
    pub fn invoke(&mut self, 
        iter: &mut TensorIteratorBase,
        gen:  RNG)  {
        
        todo!();
        /*
            random_kernel(iter, gen);
        */
    }
}


// ====================================================================================================================

pub fn uniform_and_transform<Scalar, accscalar_t, const curand4_engine_calls: Size, RNG, transform_t>(
        iter:      &mut TensorIteratorBase,
        gen:       RNG,
        transform: Transform)  {

    todo!();
        /*
            if (is_same<Scalar, double>::value) {
        distribution_nullary_kernel<Scalar, accscalar_t, curand4_engine_calls/2>(iter,
          gen,
          [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform2_double(state); },
          transform);
      } else {
        distribution_nullary_kernel<Scalar, accscalar_t, curand4_engine_calls>(iter,
          gen,
          [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
          transform);
      }
        */
}

pub fn normal_and_transform<Scalar, accscalar_t, const curand4_engine_calls: Size, RNG, transform_t>(
        iter:      &mut TensorIteratorBase,
        gen:       RNG,
        transform: Transform)  {

    todo!();
        /*
            if (is_same<Scalar, double>::value) {
        distribution_nullary_kernel<Scalar, accscalar_t, curand4_engine_calls/2>(iter,
          gen,
          [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal2_double(state); },
          transform);
      } else {
        distribution_nullary_kernel<Scalar, accscalar_t, curand4_engine_calls>(iter,
          gen,
          [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
          transform);
      }
        */
}

// ==================================================== Normal ========================================================

pub fn normal_kernel<RNG>(
        self_: &mut Tensor,
        mean:  f64,
        std:   f64,
        gen:   RNG)  {

    todo!();
        /*
            auto iter = TensorIterator::borrowing_nullary_op(self);
      AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "normal_kernel_cuda", [&] {
        using accscalar_t = acc_type<Scalar, true>;
        auto mean = static_cast<accscalar_t>(mean_);
        auto std = static_cast<accscalar_t>(std_);
        // define lambda to multiply std and add mean
        auto normal_func = [mean, std] __device__ (accscalar_t rand) {
          return static_cast<Scalar>(transformation::normal<accscalar_t>(rand, mean, std));
        };
        normal_and_transform<Scalar, accscalar_t, curand4_engine_calls>(iter, gen, normal_func);
       });
        */
}

pub struct NormalKernel<RNG> {

}

impl NormalKernel<RNG> {
    
    pub fn invoke(
        &mut self, 
        self_: &mut Tensor,
        mean:  f64,
        std:   f64,
        gen:   Option<dyn GeneratorInterface>

    ) {
        
        todo!();
        /*
            normal_kernel(self, mean, std, check_generator<RNG>(gen));
        */
    }
}

// ==================================================== Uniform ========================================================

pub fn uniform_kernel<RNG>(
    iter: &mut TensorIteratorBase,
    from: f64,
    to:   f64,
    gen:  RNG

) {

    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "uniform_kernel_cuda", [&] {
        auto from = static_cast<Scalar>(from_);
        auto to = static_cast<Scalar>(to_);
        using accscalar_t = acc_type<Scalar, true>;
        auto range = static_cast<accscalar_t>(to-from);
        // define lambda to reverse bounds, multiply 'range' and add 'from_'
        auto uniform_func = [range, from] __device__ (accscalar_t rand) {
          // reverse the bounds of curand4 from (0, 1] to [0, 1)
          // Note that this method is from legacy THCTensorRandom and is likely to give
          // you more 0-s, since, the probability of gettings 1-s is higher than 0-s and
          // by reversing the bounds, we are flipping the probabilities of 1-s and 0-s.
          // BEFORE TOUCHING THIS CODE READ: https://github.com/pytorch/pytorch/issues/16706
          auto reverse_bound_rand = rand == static_cast<accscalar_t>(1.0) ? static_cast<accscalar_t>(0.0) : rand;
          return static_cast<Scalar>(reverse_bound_rand * range + from);
        };
        uniform_and_transform<Scalar, accscalar_t, curand4_engine_calls>(iter, gen, uniform_func);
       });
        */
}

pub struct UniformKernel<RNG> {

}

impl UniformKernel<RNG> {
    
    pub fn invoke(&mut self, 
        iter: &mut TensorIteratorBase,
        from: f64,
        to:   f64,
        gen:  Option<dyn GeneratorInterface>)  {
        
        todo!();
        /*
            uniform_kernel(iter, from, to, check_generator<RNG>(gen));
        */
    }
}

// ================================================== LogNormal =======================================================

pub fn log_normal_kernel<RNG>(
        iter: &mut TensorIteratorBase,
        mean: f64,
        std:  f64,
        gen:  RNG)  {

    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "log_normal_cuda", [&] {
        using accscalar_t = acc_type<Scalar, true>;
        auto mean = static_cast<accscalar_t>(mean_);
        auto std = static_cast<accscalar_t>(std_);
        // define lambda for log_normal transformation
        auto log_normal_func = [mean, std] __device__ (accscalar_t rand) {
          return static_cast<Scalar>(transformation::log_normal<accscalar_t>(transformation::normal<accscalar_t>(rand, mean, std)));
        };
        normal_and_transform<Scalar, accscalar_t, curand4_engine_calls>(iter, gen, log_normal_func);
       });
        */
}

pub struct LogNormalKernel<RNG> {

}

impl LogNormalKernel<RNG> {
    
    pub fn invoke(&mut self, 
        iter: &mut TensorIteratorBase,
        mean: f64,
        std:  f64,
        gen:  Option<dyn GeneratorInterface>)  {
        
        todo!();
        /*
            log_normal_kernel(iter, mean, std, check_generator<RNG>(gen));
        */
    }
}

// =================================================== Geometric ======================================================

pub fn geometric_kernel<RNG>(
        iter: &mut TensorIteratorBase,
        p:    f64,
        gen:  RNG)  {

    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "geometric_cuda", [&] {
        using accscalar_t = DiscreteDistributionType<Scalar>::type;
        // define lambda for geometric transformation
        auto geometric_func = [p] __device__ (accscalar_t rand) {
          return static_cast<Scalar>(transformation::geometric<accscalar_t>(rand, p));
        };
        uniform_and_transform<Scalar, accscalar_t, curand4_engine_calls>(iter, gen, geometric_func);
      });
        */
}

pub struct GeometricKernel<RNG> {

}

impl GeometricKernel<RNG> {
    
    pub fn invoke(&mut self, 
        iter: &mut TensorIteratorBase,
        p:    f64,
        gen:  Option<dyn GeneratorInterface>)  {
        
        todo!();
        /*
            geometric_kernel(iter, p, check_generator<RNG>(gen));
        */
    }
}

// ================================================== Exponential =====================================================

pub fn exponential_kernel<RNG>(
        iter:   &mut TensorIteratorBase,
        lambda: f64,
        gen:    RNG)  {

    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "exponential_cuda", [&] {
        using accscalar_t = acc_type<Scalar, true>;
        auto lambda = static_cast<accscalar_t>(lambda_);
        // define lambda for exponential transformation
        auto exponential_func = [lambda] __device__ (accscalar_t rand) {
          return static_cast<Scalar>(transformation::exponential<accscalar_t>(rand, lambda));
        };
        uniform_and_transform<Scalar, accscalar_t, curand4_engine_calls>(iter, gen, exponential_func);
       });
        */
}

pub struct ExponentialKernel<RNG> {

}

impl ExponentialKernel<RNG> {
    
    pub fn invoke(&mut self, 
        iter:   &mut TensorIteratorBase,
        lambda: f64,
        gen:    Option<dyn GeneratorInterface>)  {
        
        todo!();
        /*
            exponential_kernel(iter, lambda, check_generator<RNG>(gen));
        */
    }
}

// ==================================================== Cauchy ========================================================

pub fn cauchy_kernel<RNG>(
        iter:   &mut TensorIteratorBase,
        median: f64,
        sigma:  f64,
        gen:    RNG)  {

    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "cauchy_cuda", [&] {
        using accscalar_t = acc_type<Scalar, true>;
        auto median = static_cast<accscalar_t>(median_);
        auto sigma = static_cast<accscalar_t>(sigma_);
        // define lambda for cauchy transformation
        auto cauchy_func = [median, sigma] __device__ (accscalar_t rand) {
          return static_cast<Scalar>(transformation::cauchy<accscalar_t>(rand, median, sigma));
        };
        uniform_and_transform<Scalar, accscalar_t, curand4_engine_calls>(iter, gen, cauchy_func);
       });
        */
}

pub struct CauchyKernel<RNG> {

}

impl CauchyKernel<RNG> {
    
    pub fn invoke(&mut self, 
        iter:   &mut TensorIteratorBase,
        median: f64,
        sigma:  f64,
        gen:    Option<dyn GeneratorInterface>)  {
        
        todo!();
        /*
            cauchy_kernel(iter, median, sigma, check_generator<RNG>(gen));
        */
    }
}

// ==================================================== Bernoulli =====================================================

pub fn bernoulli_tensor_cuda_kernel<Scalar, prob_t>(
        ret:         &mut Tensor,
        p:           &Tensor,
        philox_args: PhiloxCudaState)  {

    todo!();
        /*
            auto functor = [philox_args] __device__(
              int n, Scalar& v1, Scalar& v2, Scalar& v3, Scalar& v4,
              const prob_t& p1, const prob_t& p2, const prob_t& p3, const prob_t& p4) {
            auto seeds = philox::unpack(philox_args);
            curandStatePhilox4_32_10_t state;
            curand_init(get<0>(seeds),
                        blockIdx.x * blockDim.x + threadIdx.x,
                        get<1>(seeds),
                        &state);

            // See Note [Register spilling in curand call for CUDA < 10]
            float4 rand = curand_uniform4(&state);
            switch (n) {
              case 4: {
                CUDA_KERNEL_ASSERT(0 <= p4 && p4 <= 1);
                v4 = static_cast<Scalar>(rand.w <= p4);
                // fallthrough
              }
              case 3: {
                CUDA_KERNEL_ASSERT(0 <= p3 && p3 <= 1);
                v3 = static_cast<Scalar>(rand.z <= p3);
                // fallthrough
              }
              case 2: {
                CUDA_KERNEL_ASSERT(0 <= p2 && p2 <= 1);
                v2 = static_cast<Scalar>(rand.y <= p2);
                // fallthrough
              }
              case 1: {
                CUDA_KERNEL_ASSERT(0 <= p1 && p1 <= 1);
                v1 = static_cast<Scalar>(rand.x <= p1);
              }
            }
          };
      // The template argument `4` below indicates that we want to operate on four
      // element at each time. See NOTE [ CUDA_tensor_applyN helpers ] for details.
      CUDA_tensor_apply2<Scalar, prob_t, 4, decltype(functor),
                                   /*max_threads_per_block=*/512,
                                   /*min_blocks_per_sm==*/2>(ret, p, functor);
        */
}

pub fn bernoulli_kernel_with_tensor<RNG>(
        self_: &mut Tensor,
        p:     &Tensor,
        gen:   RNG)  {

    todo!();
        /*
            PhiloxCudaState rng_engine_inputs;
      {
        // See Note [Acquire lock when using random generators]
        lock_guard<mutex> lock(gen->mutex_);
        rng_engine_inputs = gen->philox_cuda_state(10);
      }
      auto p_CUDA = p_.to(kCUDA);
      MaybeOwned<Tensor> p = expand_inplace(self, p_CUDA);
      AT_DISPATCH_ALL_TYPES_AND3(
        ScalarType::Half, ScalarType::BFloat16, ScalarType::Bool, self.scalar_type(), "bernoulli_tensor_cuda_self_", [&] {
          using self_t = Scalar;
          AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, p->scalar_type(), "bernoulli_tensor_cuda_p_", [&] {
            using p_t = Scalar;
            return bernoulli_tensor_cuda_kernel<self_t, p_t>(self, *p, rng_engine_inputs);
          });
       });
        */
}

pub fn bernoulli_kernel<RNG>(
        iter: &mut TensorIteratorBase,
        p:    f64,
        gen:  RNG)  {

    todo!();
        /*
            AT_DISPATCH_ALL_TYPES_AND3(
        ScalarType::Half, ScalarType::BFloat16, ScalarType::Bool, iter.dtype(), "bernoulli_scalar_cuda_", [&] {
          using accscalar_t = DiscreteDistributionType<Scalar>::type;
          // define lambda for bernoulli transformation
          auto bernoulli_func = [p] __device__ (accscalar_t rand) {
            return static_cast<Scalar>(transformation::bernoulli<accscalar_t>(rand, p));
          };
          uniform_and_transform<Scalar, accscalar_t, curand4_engine_calls>(iter, gen, bernoulli_func);
       });
        */
}

pub struct BernoulliKernel<RNG> {

}

impl BernoulliKernel<RNG> {
    
    pub fn invoke(&mut self, 
        iter: &mut TensorIteratorBase,
        p:    f64,
        gen:  Option<dyn GeneratorInterface>)  {
        
        todo!();
        /*
            bernoulli_kernel(iter, p, check_generator<RNG>(gen));
        */
    }
    
    pub fn invoke(
        &mut self, 
        self_: &mut Tensor,
        p:     &Tensor,
        gen:   Option<dyn GeneratorInterface>

    ) {
        
        todo!();
        /*
            bernoulli_kernel(self, p_, check_generator<RNG>(gen));
        */
    }
}
