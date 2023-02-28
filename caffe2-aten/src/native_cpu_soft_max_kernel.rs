crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/SoftMaxKernel.cpp]

/**
  | [Note AVX-SSE transitions] In general we avoid
  | calls into cmath for code compiled with
  | AVX/AVX2 This is because of SSE-AVX transitions
  | and a bug in Glibc2.23 See
  | https://bugs.launchpad.net/ubuntu/+source/glibc/+bug/1663280
  |
  | On grainsize: The grainsize is chosen to
  | roughly get GRAIN_SIZE number of computations
  | per task. Each task works across dim_size
  | elements. 16 should be a very rough
  | approximation of the number of computations per
  | dim_size element by counting simple
  | computations (*, +, -) as 1 and exp or log as
  | 4.
  |
  */
#[inline] pub fn vec_log_softmax_lastdim<Scalar>(
        input_data_base:  *mut Scalar,
        output_data_base: *mut Scalar,
        outer_size:       i64,
        dim_size:         i64)  {

    todo!();
        /*
            using Vec = vec::Vectorized<Scalar>;
      static constexpr i64 CHUNK_SIZE = (128 / sizeof(Scalar)) * Vec::size();
      i64 grain_size = internal::GRAIN_SIZE / (16 * dim_size * CHUNK_SIZE);
      if (grain_size < CHUNK_SIZE)
        grain_size = CHUNK_SIZE;

      parallel_for(
          0,
          outer_size,
          grain_size,
          [&](i64 begin, i64 end) {
            for (i64 ii = begin; ii < end; ii += CHUNK_SIZE) {
              Scalar tmp_sum_scalar[CHUNK_SIZE];
              Scalar max_input_arr[CHUNK_SIZE];
              i64 loop_end = CHUNK_SIZE;
              if (ii + CHUNK_SIZE > end)
                loop_end = end - ii;
              for (i64 j = 0; j < loop_end; j++) {
                i64 i = ii + j;
                Scalar* input_data = input_data_base + i * dim_size;
                max_input_arr[j] = vec::reduce_all<Scalar>(
                    [](Vec& x, Vec& y) { return vec::maximum(x, y); },
                    input_data,
                    dim_size);
              }
              for (i64 j = 0; j < loop_end; j++) {
                i64 i = ii + j;
                Scalar* input_data = input_data_base + i * dim_size;
                Scalar max_input = max_input_arr[j];
                tmp_sum_scalar[j] = vec::map_reduce_all<Scalar>(
                    [max_input](Vec x) { return (x - Vec(max_input)).exp(); },
                    [](Vec x, Vec y) { return x + y; },
                    input_data,
                    dim_size);
              }
              // See [Note AVX-SSE transitions] for why this should call the
              // vectorized version (aside from perf improvements).
              vec::map(
                  [](Vec x) { return x.log(); },
                  tmp_sum_scalar,
                  tmp_sum_scalar,
                  loop_end);
              for (i64 j = 0; j < loop_end; j++) {
                i64 i = ii + j;
                Scalar* input_data = input_data_base + i * dim_size;
                Scalar* output_data = output_data_base + i * dim_size;
                Scalar tmp_sum = tmp_sum_scalar[j];
                Scalar max_input = max_input_arr[j];

                // It's necessary to keep the order of the operations below.
                // In some cases that input is large digits and the difference
                // is small, if we compute `max_input` plus `tmp_sum` before,
                // there would be a numerical problem. See an example in
                // https://github.com/pytorch/pytorch/issues/11752#issuecomment-422883379
                vec::map(
                    [tmp_sum, max_input](Vec x) { return x - Vec(max_input) - Vec(tmp_sum); },
                    output_data,
                    input_data,
                    dim_size);
              }
            }
          });
        */
}

#[inline] pub fn vec_softmax_lastdim<Scalar>(
        input_data_base:  *mut Scalar,
        output_data_base: *mut Scalar,
        outer_size:       i64,
        dim_size:         i64)  {

    todo!();
        /*
            using Vec = vec::Vectorized<Scalar>;
      i64 grain_size = internal::GRAIN_SIZE / (16 * dim_size);
      if (grain_size < 1)
        grain_size = 1;

      parallel_for(
          0,
          outer_size,
          grain_size,
          [&](i64 begin, i64 end) {
            for (i64 i = begin; i < end; i++) {
              Scalar* input_data = input_data_base + i * dim_size;
              Scalar* output_data = output_data_base + i * dim_size;
              Scalar max_input = vec::reduce_all<Scalar>(
                  [](Vec& x, Vec& y) { return vec::maximum(x, y); },
                  input_data,
                  dim_size);
              vec::map(
                  [max_input](Vec x) { return (x - Vec(max_input)).exp(); },
                  output_data,
                  input_data,
                  dim_size);
              Scalar tmp_sum = vec::reduce_all<Scalar>(
                  [](Vec x, Vec y) { return x + y; }, output_data, dim_size);
              tmp_sum = 1 / tmp_sum;
              vec::map(
                  [tmp_sum](Vec x) { return x * Vec(tmp_sum); },
                  output_data,
                  output_data,
                  dim_size);
            }
          });
        */
}

#[inline] pub fn vec_host_softmax_backward_lastdim<Scalar, const log_softmax: bool>(
        grad_input_data_base: *mut Scalar,
        grad_data_base:       *mut Scalar,
        output_data_base:     *mut Scalar,
        outer_size:           i64,
        dim_size:             i64)  {

    todo!();
        /*
            using Vec = vec::Vectorized<Scalar>;
      i64 grain_size = internal::GRAIN_SIZE / (16 * dim_size);
      if (grain_size < 1)
        grain_size = 1;

      parallel_for(
          0,
          outer_size,
          grain_size,
          [&](i64 begin, i64 end) {
            for (i64 i = begin; i < end; i++) {
              Scalar* grad_input_data = grad_input_data_base + i * dim_size;
              Scalar* grad_data = grad_data_base + i * dim_size;
              Scalar* output_data = output_data_base + i * dim_size;
              Scalar sum;
              if (log_softmax) {
                sum = vec::reduce_all<Scalar>(
                    [](Vec& x, Vec& y) { return x + y; }, grad_data, dim_size);
              } else {
                sum = vec::map2_reduce_all<Scalar>(
                    [](Vec x, Vec y) { return x * y; },
                    [](Vec x, Vec y) { return x + y; },
                    grad_data,
                    output_data,
                    dim_size);
              }
              if (log_softmax) {
                vec::map2(
                    [sum](Vec x, Vec y) { return x - ((y.exp()) * Vec(sum)); },
                    grad_input_data,
                    grad_data,
                    output_data,
                    dim_size);
              } else {
                vec::map2(
                    [sum](Vec x, Vec y) { return (x - Vec(sum)) * y; },
                    grad_input_data,
                    grad_data,
                    output_data,
                    dim_size);
              }
            }
          });
        */
}

pub struct VecHostSoftMaxLastDim<Scalar,const LogSoftMax: bool> {

}

impl<Scalar, const LOG_SOFTMAX: bool> VecHostSoftMaxLastDim<Scalar, LOG_SOFTMAX> {
    
    pub fn apply(
        output: &mut Tensor,
        input:  &Tensor)  {
        
        todo!();
        /*
            i64 outer_size = 1;
        i64 dim_size = input.size(input.ndimension() - 1);
        for (i64 i = 0; i < input.ndimension() - 1; ++i)
          outer_size *= input.size(i);
        Scalar* input_data_base = input.data_ptr<Scalar>();
        Scalar* output_data_base = output.data_ptr<Scalar>();
        if (LogSoftMax) {
          _vec_log_softmax_lastdim(
              input_data_base, output_data_base, outer_size, dim_size);
        } else {
          _vec_softmax_lastdim(
              input_data_base, output_data_base, outer_size, dim_size);
        }
        */
    }
}

#[inline] pub fn vec_softmax<Scalar>(
        input_data_base:  *mut Scalar,
        output_data_base: *mut Scalar,
        outer_size:       i64,
        inner_size:       i64,
        dim_size:         i64)  {

    todo!();
        /*
            using Vec = vec::Vectorized<Scalar>;
      i64 dim_stride = inner_size;
      i64 outer_stride = dim_size * dim_stride;
      i64 grain_size = min(internal::GRAIN_SIZE / dim_size, (i64)1);
      int vectorized_step = Vec().size(); // Currently, we only support Scalar with double or float32
      TORCH_CHECK(
        (vectorized_step == 8) || (vectorized_step == 4),
        "vectorized_step must be 8 with dtype float or 4 with dtype double");
      parallel_for(
          0, outer_size * inner_size, grain_size, [&](i64 begin, i64 end) {
            i64 idx = begin;
            while (idx < end) {
              i64 outer_idx = idx / inner_size;
              i64 inner_idx = idx % inner_size;
              if (((inner_idx + vectorized_step) <= inner_size) && ((idx + vectorized_step) <= end)) {
                // Vectorization
                Scalar* input_data =
                    input_data_base + outer_idx * outer_stride + inner_idx;
                Scalar* output_data =
                    output_data_base + outer_idx * outer_stride + inner_idx;
                // Step 1: Get max Score
                Vec max_m256 = Vec::loadu(input_data);
                for (i64 d = 1; d < dim_size; d += 1) {
                  Vec input_m256 = Vec::loadu(input_data + d * dim_stride);
                  max_m256 = vec::maximum(max_m256, input_m256);
                }
                // Step2: Calculate sum
                Vec sum_m256 = Vec(0.0);
                for (i64 d = 0; d < dim_size; d += 1) {
                  Vec output_m256 =
                      (Vec::loadu(input_data + d * dim_stride) - max_m256).exp();
                  output_m256.store(output_data + d * dim_stride);
                  sum_m256 = sum_m256 + output_m256;
                }
                // Step3: Unify
                for (i64 d = 0; d < dim_size; d += 1) {
                  Vec output_m256 =
                      Vec::loadu(output_data + d * dim_stride) / sum_m256;
                  output_m256.store(output_data + d * dim_stride);
                }
                idx += vectorized_step;
              } else {
                // Tail case(Scalar): it is exactly same logic as host_softmax
                // inside aten/src/ATen/native/SoftMax.cpp. There are 2 kind of
                // cases which will fall through this part:
                // Case 1: For the idx at the end of total chunk for each thread, there are not enough numbers for parallization.
                // Case 2: For the idx at the end of each inner_size inside thread, there are not enough numbers for parallization.
                i64 tail_number = ((idx+vectorized_step) > end) ? /*Case1*/ (end - idx) : /*Case2*/ (inner_size - inner_idx);
                for (i64 i=0; i < tail_number; i++) {
                  outer_idx = (idx + i) / inner_size;
                  inner_idx = (idx + i) % inner_size;
                  Scalar* input_data =
                      input_data_base + outer_idx * outer_stride + inner_idx;
                  Scalar* output_data =
                      output_data_base + outer_idx * outer_stride + inner_idx;
                  // Step1: Get max score
                  Scalar max_input = input_data[0];
                  for (i64 d = 1; d < dim_size; d += 1) {
                    max_input = max(max_input, input_data[d * dim_stride]);
                  }
                  // Step2: Calculate the Sum
                  Scalar sum_data = 0;
                  for (i64 d = 0; d < dim_size; d += 1) {
                    output_data[d * dim_stride] =
                        exp(input_data[d * dim_stride] - max_input);
                    sum_data += output_data[d * dim_stride];
                  }
                  // Step3: Unify
                  for (i64 d = 0; d < dim_size; d += 1) {
                    output_data[d * dim_stride] =
                        output_data[d * dim_stride]/sum_data;
                  }
                }
                idx += tail_number;
              }
            }
          });
        */
}

pub struct VecSoftMax<Scalar,const LogSoftMax: bool> {

}

impl<Scalar,const LOG_SOFTMAX: bool> VecSoftMax<Scalar, LOG_SOFTMAX> {

    pub fn apply(
        output: &mut Tensor,
        input:  &Tensor,
        dim:    i64)  {
        
        todo!();
        /*
            i64 outer_size = 1;
        i64 dim_size = input.size(dim);
        i64 inner_size = 1;
        for (i64 i = 0; i < dim; ++i)
          outer_size *= input.size(i);
        for (i64 i = dim + 1; i < input.dim(); ++i)
          inner_size *= input.size(i);
        Scalar* input_data_base = input.data_ptr<Scalar>();
        Scalar* output_data_base = output.data_ptr<Scalar>();
        if (LogSoftMax) {
          AT_ERROR("vec_softmax not implemented for LogSoftMax");
        } else {
          _vec_softmax(
              input_data_base, output_data_base, outer_size, inner_size, dim_size);
        }
        */
    }
}

pub struct VecHostSoftMaxBackwardLastDim<Scalar,const LOG_SOFTMAX: bool> {

}

impl<Scalar, const LOG_SOFTMAX: bool> VecHostSoftMaxBackwardLastDim<Scalar, LOG_SOFTMAX> {
    
    pub fn apply(
        grad_input: &mut Tensor,
        grad:       &Tensor,
        output:     &Tensor)  {
        
        todo!();
        /*
            i64 outer_size = 1;
        i64 dim_size = grad.size(grad.ndimension() - 1);
        for (i64 i = 0; i < grad.ndimension() - 1; ++i)
          outer_size *= grad.size(i);
        Scalar* grad_input_data_base = grad_input.data_ptr<Scalar>();
        Scalar* grad_data_base = grad.data_ptr<Scalar>();
        Scalar* output_data_base = output.data_ptr<Scalar>();
        _vec_host_softmax_backward_lastdim<Scalar, LogSoftMax>(
            grad_input_data_base,
            grad_data_base,
            output_data_base,
            outer_size,
            dim_size);
        */
    }
}

pub fn softmax_lastdim_kernel_impl(
        result: &mut Tensor,
        self_:  &Tensor)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "softmax_lastdim_kernel_impl", [&] {
        vec_host_softmax_lastdim<Scalar, false>::apply(result, self);
      });
        */
}

pub fn softmax_kernel_impl(
        result: &mut Tensor,
        self_:  &Tensor,
        dim:    i64)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "softmax_kernel_impl", [&] {
        vec_softmax<Scalar, false>::apply(result, self, dim);
      });
        */
}

pub fn log_softmax_lastdim_kernel_impl(
        result: &mut Tensor,
        self_:  &Tensor)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND(
          ScalarType::BFloat16, self.scalar_type(),
          "log_softmax_lastdim_kernel_impl",
          [&] { vec_host_softmax_lastdim<Scalar, true>::apply(result, self); });
        */
}

pub fn softmax_backward_lastdim_kernel_impl(
        grad_input: &mut Tensor,
        grad:       &Tensor,
        output:     &Tensor)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(
          grad.scalar_type(), "softmax_backward_lastdim_kernel_impl", [&] {
            vec_host_softmax_backward_lastdim<Scalar, false>::apply(
                grad_input, grad, output);
          });
        */
}

pub fn log_softmax_backward_lastdim_kernel_impl(
        grad_input: &mut Tensor,
        grad:       &Tensor,
        output:     &Tensor)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND(
          ScalarType::BFloat16, grad.scalar_type(),
          "log_softmax_backward_lastdim_kernel_impl", [&] {
            vec_host_softmax_backward_lastdim<Scalar, true>::apply(
                grad_input, grad, output);
          });
        */
}

register_dispatch!{softmax_lastdim_kernel              , &softmax_lastdim_kernel_impl}
register_dispatch!{log_softmax_lastdim_kernel          , &log_softmax_lastdim_kernel_impl}
register_dispatch!{softmax_backward_lastdim_kernel     , &softmax_backward_lastdim_kernel_impl}
register_dispatch!{log_softmax_backward_lastdim_kernel , &log_softmax_backward_lastdim_kernel_impl}
register_dispatch!{softmax_kernel                      , &softmax_kernel_impl}
