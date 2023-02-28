crate::ix!();

use crate::{
    Tensor,
    OperatorStorage,
    Operator,
    CPUContext,
    OperatorDef,
    Workspace,
};

#[inline] pub fn compute_square_average_inlined(a: *const f32, len: i32) -> f32 {
    
    todo!();
    /*
        float sum = 0.0f;

      int i = 0;
    #ifdef __AVX__
      constexpr int kSize = 8;
      __m256 partial_sum = _mm256_setzero_ps();
      for (; i + kSize <= len; i += kSize) {
        __m256 ai = _mm256_loadu_ps(a + i);
        partial_sum = _mm256_add_ps(partial_sum, _mm256_mul_ps(ai, ai));
      }
      // Reduce sum to 1 value
      __m256 partial_sum_2 = _mm256_hadd_ps(partial_sum, partial_sum);
      __m256 partial_sum_3 = _mm256_hadd_ps(partial_sum_2, partial_sum_2);
      sum = _mm_cvtss_f32(_mm256_castps256_ps128(partial_sum_3)) +
          _mm_cvtss_f32(_mm256_extractf128_ps(partial_sum_3, 1));
    #endif

      for (; i < len; ++i) {
        sum = std::fma(a[i], a[i], sum);
      }

      return sum / len;
    */
}

#[inline] pub fn compute_square_average_with_weight_decay_inlined(
    a:            *const f32,
    w:            *const f32,
    len:          i32,
    weight_decay: f32) -> f32 
{
    todo!();
    /*
        float sum = 0.0f;

      int i = 0;
    #ifdef __AVX__
      constexpr int kSize = 8;
      __m256 partial_sum = _mm256_setzero_ps();
      __m256 weight_decay_v = _mm256_set1_ps(weight_decay);
      for (; i + kSize <= len; i += kSize) {
        __m256 ai = _mm256_loadu_ps(a + i);
        __m256 wi = _mm256_loadu_ps(w + i);
    #ifdef __FMA__
        ai = _mm256_fmadd_ps(weight_decay_v, wi, ai);
    #else
        ai = _mm256_add_ps(_mm256_mul_ps(weight_decay_v, wi), ai);
    #endif
        partial_sum = _mm256_add_ps(partial_sum, _mm256_mul_ps(ai, ai));
      }
      // Reduce sum to 1 value
      __m256 partial_sum_2 = _mm256_hadd_ps(partial_sum, partial_sum);
      __m256 partial_sum_3 = _mm256_hadd_ps(partial_sum_2, partial_sum_2);
      sum = _mm_cvtss_f32(_mm256_castps256_ps128(partial_sum_3)) +
          _mm_cvtss_f32(_mm256_extractf128_ps(partial_sum_3, 1));
    #endif

      for (; i < len; ++i) {
        float ai = std::fma(weight_decay, w[i], a[i]);
        sum = std::fma(ai, ai, sum);
      }

      return sum / len;
    */
}

#[inline] pub fn compute_square_average_with_weight_decay_inlined_f16_weights(
    a:            *const f32,
    w:            *const f16,
    len:          i32,
    weight_decay: f32) -> f32 
{
    todo!();
    /*
        float sum = 0.0f;

      int i = 0;
    #ifdef __AVX__
      constexpr int kSize = 8;
      __m256 partial_sum = _mm256_setzero_ps();
      __m256 weight_decay_v = _mm256_set1_ps(weight_decay);
      for (; i + kSize <= len; i += kSize) {
        __m256 ai = _mm256_loadu_ps(a + i);
        __m128i whi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i));
        __m256 wi = _mm256_cvtph_ps(whi);
    #ifdef __FMA__
        ai = _mm256_fmadd_ps(weight_decay_v, wi, ai);
    #else
        ai = _mm256_add_ps(_mm256_mul_ps(weight_decay_v, wi), ai);
    #endif
        partial_sum = _mm256_add_ps(partial_sum, _mm256_mul_ps(ai, ai));
      }
      // Reduce sum to 1 value
      __m256 partial_sum_2 = _mm256_hadd_ps(partial_sum, partial_sum);
      __m256 partial_sum_3 = _mm256_hadd_ps(partial_sum_2, partial_sum_2);
      sum = _mm_cvtss_f32(_mm256_castps256_ps128(partial_sum_3)) +
          _mm_cvtss_f32(_mm256_extractf128_ps(partial_sum_3, 1));
    #endif

      for (; i < len; ++i) {
        float ai = std::fma(weight_decay, w[i], a[i]);
        sum = std::fma(ai, ai, sum);
      }

      return sum / len;
    */
}

/**
  | Fused operator of
  | 
  | SparseLengthsIndicesInGradientSumGradient
  | (gradient of SparseLengthsSum) + RowWiseSparseAdagrad.
  | 
  | BW saving analysis for numSegments
  | B, L_avg = avg(lengths), block_size
  | D, assuming T = float and SIndex = int64_t:
  | 
  | Before fusion,
  | 
  | SparseLengthsIndicesInGradientSumGradient
  | reads
  | 
  | B*D*4 and writes
  | 
  | B*L_avg*D*4. RowWiseSparseAdagrad
  | reads
  | 
  | B*2*L_avg*D*4 and writes B*L_avg*D*4.
  | So, the total memory traffic is B*(1+4*L_avg)*D*4.
  | 
  | After fusion, we read B*(1+L_avg)*D*4
  | and write
  | 
  | B*L_avg*D*4 with total memory traffic
  | 
  | B*(1+2*L_avg)*D*4.
  | 
  | Assuming L_avg >> 1, the memory BW is
  | saving is about 2x .
  | 
  | See https://fb.quip.com/ldG7A55Ur5wM
  | for more details on BW saving analysis
  | and evaluation results. 
  |
  | typename Tdata, // embedding types 
  | typename T,     // everything else
  | 
  | Fused operator of
  | 
  | SparseLengthsIndicesInGradientSumGradient
  | (gradient of SparseLengthsSum) + RowWiseSparseAdagrad.
  | 
  | Given inputs (param, moment, indices,
  | grad, lr), runs the row-wise sparse
  | AdaGrad update on (param, grad, moment[indices],
  | lr), and returns (new_param, new_moment)
  | as in the dense case. Additional input
  | (lengths) is for fused
  | 
  | SparseLengthsSumGradient operator.
  |
  */
pub struct RowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp<
Tdata,
T,
TLengths,
RowWiseAdagradT,
const is_mean: bool = false> 
{
    storage: OperatorStorage,
    context: CPUContext,

    epsilon:      T,
    weight_decay: T,
    kernel:       RowWiseAdagradT,
    grad_buffer:  Tensor, // default = CPU

    phantomA: PhantomData<Tdata>,
    phantomB: PhantomData<TLengths>,
}

num_inputs!{RowWiseSparseAdagradFusedWithSparseLengthsSumGradient, 6}

num_outputs!{RowWiseSparseAdagradFusedWithSparseLengthsSumGradient, 2}

inputs!{RowWiseSparseAdagradFusedWithSparseLengthsSumGradient, 
    0 => ("param",    "Parameters to be updated"),
    1 => ("moment",   "Moment history"),
    2 => ("indices",  "Integer vector containing indices of the first dimension of param for the slices that are being updated"),
    3 => ("grad",     "Gradient computed"),
    4 => ("lr",       "learning rate"),
    5 => ("lengths",  "Non negative vector with sum of elements equal to indices length")
}

outputs!{RowWiseSparseAdagradFusedWithSparseLengthsSumGradient, 
    0 => ("output_param",  "Updated parameters"),
    1 => ("output_moment", "Updated moment")
}

args!{RowWiseSparseAdagradFusedWithSparseLengthsSumGradient, 
    0 => ("round_option", "rounding option: 0 for nearest rounding, 1 for stochastic rounding"),
    1 => ("epsilon",      "Default 1e-5")
}

enforce_one_to_one_inplace!{RowWiseSparseAdagradFusedWithSparseLengthsSumGradient}

register_cpu_operator!{
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradient,
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp::<
        f32,
        f32,
        i32,
        rowwise_adagrad_update_inlined,
        IsMeanFlag>
}

register_cpu_operator_with_engine!{
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradient,
    SIMD,
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp::<
        f32,
        f32,
        i32,
        rowwise_adagrad_update_inlined,
        IsMeanFlag>
}

input_tags!{
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp
    {
        Param,
        Moment1,
        Indices,
        Grad,
        Lr,
        Lengths
    }
}

output_tags!{
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp
    {
        OutputParam,
        OutputMoment1
    }
}

impl<Tdata, T, TLengths, RowWiseAdagradT, const is_mean: bool> 
RowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp<
Tdata, T, TLengths, RowWiseAdagradT, is_mean> 
{
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws),
            epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5)),
            weight_decay_( this->template GetSingleArgument<float>("weight_decay", 0.f)) 

        VLOG(1) << "gradient optimization operator in use: "
                << "RowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp";
        const T decay = this->template GetSingleArgument<T>("decay", 1.0);
        CAFFE_ENFORCE_EQ(
            decay, 1.0, "Decay is not supported for SparseSimdAdagradOp");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // Enforce shapes
        CAFFE_ENFORCE_EQ(Input(PARAM).sizes()[0], Input(MOMENT_1).numel());
        CAFFE_ENFORCE_EQ(Input(LR).numel(), 1);
        CAFFE_ENFORCE_EQ(
            Input(PARAM).size_from_dim(1),
            Input(GRAD).size_from_dim(Input(INDICES).dim()));

        return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, Input(INDICES));
        */
    }
    
    #[inline] pub fn do_run_with_type<SIndex>(&mut self) -> bool {
        todo!();
        /*
            const auto* lr = Input(LR).template data<T>();
        Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
        Output(OUTPUT_MOMENT_1)->ResizeLike(Input(MOMENT_1));

        auto& segmentGradsInput = Input(GRAD);
        auto& lengthsInput = Input(LENGTHS);

        CAFFE_ENFORCE_EQ(lengthsInput.dim(), 1, "LENGTHS must be a vector");
        auto numSegments = lengthsInput.size(0);
        CAFFE_ENFORCE_GT(segmentGradsInput.dim(), 0);
        CAFFE_ENFORCE_EQ(numSegments, segmentGradsInput.size(0));
        const auto* lengths = lengthsInput.template data<TLengths>();

        auto n = Input(INDICES).numel();
        auto numParams = Input(PARAM).numel();

        const auto* indices = Input(INDICES).template data<SIndex>();
        const auto* gradIn = segmentGradsInput.template data<T>();
        const auto* paramIn = Input(PARAM).template data<Tdata>();
        const auto* momentIn = Input(MOMENT_1).template data<T>();

        auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<Tdata>();
        auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<T>();

        if (numSegments == 0) {
          return true;
        }

        auto block_size = segmentGradsInput.size_from_dim(1);

        // Enforce:
        // Input(embedding/momentum) == outputs(embedding/momentum)
        CAFFE_ENFORCE_EQ(
            Input(PARAM).numel() / block_size,
            Input(MOMENT_1).numel(),
            "Input Param size: ",
            Input(PARAM).numel(),
            " Block size: ",
            block_size,
            " Input Moment size: ",
            Input(MOMENT_1).numel());

        if (is_mean) {
          grad_buffer_.ResizeLike(Input(GRAD));
        }
        auto* grad_buffer_data =
            is_mean ? grad_buffer_.template mutable_data<T>() : NULL;
        if (is_mean) {
          for (auto rangeIndex = 0; rangeIndex < numSegments; ++rangeIndex) {
            for (auto tmpIndex = 0; tmpIndex < block_size; ++tmpIndex) {
              auto offsetI = rangeIndex * block_size;
              grad_buffer_data[offsetI + tmpIndex] = lengths[rangeIndex] > 0
                  ? gradIn[offsetI + tmpIndex] / lengths[rangeIndex]
                  : gradIn[offsetI + tmpIndex];
            }
          }
        }

        compute<SIndex>(
            block_size,
            indices,
            n,
            lengths,
            numSegments,
            is_mean ? grad_buffer_data : gradIn,
            paramIn,
            numParams,
            momentIn,
            paramOut,
            momentOut,
            epsilon_,
            lr[0],
            weight_decay_,
            kernel_);

        return true;
        */
    }
    
    #[inline] fn compute_impl<SIndex, const HAS_WEIGHT_DECAY: bool>(
        block_size:   i64,
        indices:      *const SIndex,
        n:            i64,
        lengths:      *const TLengths,
        num_segments: i64,
        grad_in:      *const T,
        param_in:     *const Tdata,
        num_params:   i64,
        moment_in:    *const T,
        param_out:    *mut Tdata,
        moment_out:   *mut T,
        epsilon:      f32,
        lr:           T,
        weight_decay: T,
        kernel:       &mut RowWiseAdagradT)  {
        todo!();
        /*
            int dataIndex = 0;
        for (auto rangeIndex = 0; rangeIndex < numSegments; ++rangeIndex) {
          auto offsetI = rangeIndex * block_size;
          const float* g = gradIn + offsetI;

          float g_sq_avg = 0;
          if (block_size > 1 && !HAS_WEIGHT_DECAY) {
            g_sq_avg = internal::compute_square_average_inlined_(g, block_size);
          }

          for (auto start = dataIndex; dataIndex < start + lengths[rangeIndex];
               ++dataIndex) {
            std::size_t idx = indices[dataIndex];
            auto offsetIdx = idx * block_size;

            // Enforce:
            // access within range
            // gradient access within range
            CAFFE_ENFORCE_GE(
                numParams,
                block_size + offsetIdx,
                "Accessing params out of bound,  idx:",
                idx,
                " for input dataIndex:",
                dataIndex,
                " and block size:",
                block_size,
                " max size:",
                numParams);

            if (block_size == 1) {
              float gi = std::fma(weight_decay, paramIn[idx], *g);
              float hi = momentOut[idx] = momentIn[idx] + gi * gi;
              paramOut[idx] = paramIn[idx] + lr / (std::sqrt(hi) + epsilon) * gi;
            } else {
              // prefetching
              const int prefdist_T0 = 16;
              int i_pref = (dataIndex < n - prefdist_T0) ? dataIndex + prefdist_T0
                                                         : dataIndex;
              std::size_t idx_pref = indices[i_pref];

              if (HAS_WEIGHT_DECAY) {
                g_sq_avg =
                    internal::compute_square_average_with_weight_decay_inlined_(
                        g, paramOut + offsetIdx, block_size, weight_decay);
              }

              kernel(
                  block_size,

                  paramOut + offsetIdx,
                  &paramOut[idx_pref * block_size],

                  g,
                  g_sq_avg,

                  momentOut + idx,
                  momentOut + idx_pref,

                  epsilon,
                  lr,
                  HAS_WEIGHT_DECAY ? weight_decay : 0.0f);
            }
          }
        }
        CAFFE_ENFORCE_EQ(dataIndex, n);
        */
    }
    
    #[inline] pub fn compute<SIndex>(
        block_size:   i64,
        indices:      *const SIndex,
        n:            i64,
        lengths:      *const TLengths,
        num_segments: i64,
        grad_in:      *const T,
        param_in:     *const Tdata,
        num_params:   i64,
        moment_in:    *const T,
        param_out:    *mut Tdata,
        moment_out:   *mut T,
        epsilon:      f32,
        lr:           T,
        weight_decay: T,
        kernel:       &mut RowWiseAdagradT)  {
        todo!();
        /*
            if (weight_decay == 0.0f) {
          compute<SIndex, false>(
              block_size,
              indices,
              n,
              lengths,
              numSegments,
              gradIn,
              paramIn,
              numParams,
              momentIn,
              paramOut,
              momentOut,
              epsilon,
              lr,
              0.0f,
              kernel);
        } else {
          compute<SIndex, true>(
              block_size,
              indices,
              n,
              lengths,
              numSegments,
              gradIn,
              paramIn,
              numParams,
              momentIn,
              paramOut,
              momentOut,
              epsilon,
              lr,
              weight_decay,
              kernel);
        }
        */
    }
}

/**
  | typename Tdata, // embedding types
  | typename T, // everything else
  |
  */
pub struct RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientOp<Tdata,T,TLengths,RowWiseAdagradT> {

    storate:      OperatorStorage,
    context:      CPUContext,

    epsilon:      T,
    weight_decay: T,
    kernel:       RowWiseAdagradT,

    phantomA: PhantomData<Tdata>,
    phantomB: PhantomData<TLengths>,
}

input_tags!{
    RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientOp
    {
        Param,
        Moment1,
        AuxParam,
        Indices,
        Grad,
        Lr,
        Lengths
    }
}

output_tags!{
    RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientOp
    {
        OutputParam,
        OutputMoment1,
        AuxGrad
    }
}

impl< Tdata, T, TLengths, RowWiseAdagradT> 
RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientOp<Tdata, T, TLengths, RowWiseAdagradT> 
{
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws),
            epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5)),
            weight_decay_( this->template GetSingleArgument<float>("weight_decay", 0.f)) 

        VLOG(1)
            << "gradient optimization operator in use: "
            << "RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientOp";
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // Enforce shapes
        CAFFE_ENFORCE_EQ(Input(PARAM).sizes()[0], Input(MOMENT_1).numel());
        CAFFE_ENFORCE_EQ(Input(LR).numel(), 1);
        CAFFE_ENFORCE_EQ(
            Input(PARAM).size_from_dim(1),
            Input(GRAD).size_from_dim(Input(INDICES).dim()));

        return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, Input(INDICES));
        */
    }
    
    #[inline] pub fn do_run_with_type<SIndex>(&mut self) -> bool {
        todo!();
        /*
            const auto* lr = Input(LR).template data<T>();
        Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
        Output(OUTPUT_MOMENT_1)->ResizeLike(Input(MOMENT_1));

        auto& segmentGradsInput = Input(GRAD);
        auto& lengthsInput = Input(LENGTHS);

        CAFFE_ENFORCE_EQ(lengthsInput.dim(), 1, "LENGTHS must be a vector");
        auto numSegments = lengthsInput.size(0);
        CAFFE_ENFORCE_GT(segmentGradsInput.dim(), 0);
        CAFFE_ENFORCE_EQ(numSegments, segmentGradsInput.size(0));
        const auto* lengths = lengthsInput.template data<TLengths>();

        auto n = Input(INDICES).numel();
        auto numParams = Input(PARAM).numel();

        const auto* indices = Input(INDICES).template data<SIndex>();
        const auto* gradIn = segmentGradsInput.template data<T>();
        const auto* paramIn = Input(PARAM).template data<Tdata>();
        const auto* momentIn = Input(MOMENT_1).template data<T>();
        const auto* auxParamIn = Input(AUX_PARAM).template data<T>();

        auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<Tdata>();
        auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<T>();
        Output(AUX_GRAD)->Resize(n);
        auto* auxGrad = Output(AUX_GRAD)->template mutable_data<T>();

        CAFFE_ENFORCE_EQ(
            paramIn, paramOut, "RowWiseSparseAdagrad must use inplace param");
        CAFFE_ENFORCE_EQ(
            momentIn, momentOut, "RowWiseSparseAdagrad must use inplace momentum");

        if (numSegments == 0) {
          return true;
        }

        auto block_size = segmentGradsInput.size_from_dim(1);

        // Enforce:
        // Input(embedding/momentum) == outputs(embedding/momentum)
        CAFFE_ENFORCE_EQ(
            Input(PARAM).numel() / block_size,
            Input(MOMENT_1).numel(),
            "Input Param size: ",
            Input(PARAM).numel(),
            " Block size: ",
            block_size,
            " Input Moment size: ",
            Input(MOMENT_1).numel());

        compute<SIndex>(
            block_size,
            indices,
            n,
            lengths,
            numSegments,
            gradIn,
            paramIn,
            numParams,
            momentIn,
            auxParamIn,
            paramOut,
            momentOut,
            auxGrad,
            epsilon_,
            lr[0],
            weight_decay_,
            kernel_,
            &context_);

        return true;
        */
    }
    
    #[inline] pub fn compute_impl<SIndex, const HAS_WEIGHT_DECAY: bool>(
        block_size:   i64,
        indices:      *const SIndex,
        n:            i64,
        lengths:      *const TLengths,
        num_segments: i64,
        grad_in:      *const T,
        param_in:     *const Tdata,
        num_params:   i64,
        moment_in:    *const T,
        aux_param_in: *const T,
        param_out:    *mut Tdata,
        moment_out:   *mut T,
        aux_grad:     *mut T,
        epsilon:      f32,
        lr:           T,
        weight_decay: T,
        kernel:       &mut RowWiseAdagradT,
        context:      *mut CPUContext)  {
        todo!();
        /*
            // Cannot fuse this loop with the loop below because paramIn is updated
        // by the second loop. Specifically, there could be dataIndex1 != dataIndex2
        // s.t. indices[dataIndex1] == indices[dataIndex2], and fusing these two
        // loops would violate dependencies w.r.t.
        // paramIn[indices[dataIndex1]:block_size] The approximate version.
        // (RowWiseSparseSimdAdagradFusedWithSparseLengthsWeightedSumGradientApproxOp)
        // ignores this dependency and fuses these two loops.
        std::vector<T> temp_grad(block_size);
        int dataIndex = 0;
        for (auto rangeIndex = 0; rangeIndex < numSegments; ++rangeIndex) {
          for (auto start = dataIndex; dataIndex < start + lengths[rangeIndex];
               ++dataIndex) {
            std::size_t idx = indices[dataIndex];
            auto offsetI = rangeIndex * block_size;
            auto offsetIdx = idx * block_size;

            // Enforce:
            // access within range
            // gradient access within range
            CAFFE_ENFORCE_GE(
                numParams,
                block_size + offsetIdx,
                "Accessing params out of bound,  idx:",
                idx,
                " for input dataIndex:",
                dataIndex,
                " and block size:",
                block_size,
                " max size:",
                numParams);

            // temp_aux_grad[dataIndex] = gradIn[offsetI] dot paramIn[offsetIdx]
            internal::dot<T, Tdata, T>(
                block_size,
                gradIn + offsetI,
                paramIn + offsetIdx,
                auxGrad + dataIndex,
                context);
          }
        }
        CAFFE_ENFORCE_EQ(dataIndex, n);

        dataIndex = 0;
        for (auto rangeIndex = 0; rangeIndex < numSegments; ++rangeIndex) {
          auto offsetI = rangeIndex * block_size;
          const float* g = gradIn + offsetI;

          float g_sq_avg;
          if (block_size > 1 && !HAS_WEIGHT_DECAY) {
            g_sq_avg = internal::compute_square_average_inlined_(g, block_size);
          }

          for (auto start = dataIndex; dataIndex < start + lengths[rangeIndex];
               ++dataIndex) {
            auto idx = indices[dataIndex];
            auto offsetIdx = idx * block_size;
            auto localOffset = dataIndex - start;

            for (int i = 0; i < block_size; ++i) {
              temp_grad[i] = auxParamIn[localOffset] * g[i];
            }

            if (block_size == 1) {
              float gi = std::fma(weight_decay, paramIn[idx], temp_grad[0]);
              float hi = momentOut[idx] = momentIn[idx] + gi * gi;
              paramOut[idx] = paramIn[idx] + lr / (std::sqrt(hi) + epsilon) * gi;
            } else {
              // prefetching
              const int prefdist_T0 = 16;
              int i_pref = (dataIndex < n - prefdist_T0) ? dataIndex + prefdist_T0
                                                         : dataIndex;
              std::size_t idx_pref = indices[i_pref];

              if (HAS_WEIGHT_DECAY) {
                g_sq_avg =
                    internal::compute_square_average_with_weight_decay_inlined_(
                        temp_grad.data(),
                        paramOut + offsetIdx,
                        block_size,
                        weight_decay);
              }

              kernel(
                  block_size,

                  paramOut + offsetIdx,
                  &paramOut[idx_pref * block_size],

                  temp_grad.data(),
                  g_sq_avg *
                      (HAS_WEIGHT_DECAY
                           ? 1
                           : auxParamIn[localOffset] * auxParamIn[localOffset]),

                  momentOut + idx,
                  momentOut + idx_pref,

                  epsilon,
                  lr,
                  HAS_WEIGHT_DECAY ? weight_decay : 0.0f);
            }
          }
        }
        */
    }
    
    #[inline] pub fn compute<SIndex>(
        block_size:   i64,
        indices:      *const SIndex,
        n:            i64,
        lengths:      *const TLengths,
        num_segments: i64,
        grad_in:      *const T,
        param_in:     *const Tdata,
        num_params:   i64,
        moment_in:    *const T,
        aux_param_in: *const T,
        param_out:    *mut Tdata,
        moment_out:   *mut T,
        aux_grad:     *mut T,
        epsilon:      f32,
        lr:           T,
        weight_decay: T,
        kernel:       &mut RowWiseAdagradT,
        context:      *mut CPUContext)  {
        todo!();
        /*
            if (weight_decay == 0.0f) {
          compute<SIndex, /*HAS_WEIGHT_DECAY=*/false>(
              block_size,
              indices,
              n,
              lengths,
              numSegments,
              gradIn,
              paramIn,
              numParams,
              momentIn,
              auxParamIn,
              paramOut,
              momentOut,
              auxGrad,
              epsilon,
              lr,
              0.0f,
              kernel,
              context);
        } else {
          compute<SIndex, /*HAS_WEIGHT_DECAY=*/true>(
              block_size,
              indices,
              n,
              lengths,
              numSegments,
              gradIn,
              paramIn,
              numParams,
              momentIn,
              auxParamIn,
              paramOut,
              momentOut,
              auxGrad,
              epsilon,
              lr,
              weight_decay,
              kernel,
              context);
        }
        */
    }
}

/**
  | typename Tdata, // embedding types
  | typename T, // everything else
  |
  */
pub struct RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientApproxOp<Tdata,T,TLengths,RowWiseAdagradT> {
    storage: OperatorStorage,
    context: CPUContext,

    epsilon:      T,
    weight_decay: T,
    kernel:       RowWiseAdagradT,

    phantomA: PhantomData<Tdata>,
    phantomB: PhantomData<TLengths>,
}

input_tags!{
    RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientApproxOp
    {
        Param,
        Moment1,
        AuxParam,
        Indices,
        Grad,
        Lr,
        Lengths
    }
}

output_tags!{
    RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientApproxOp
    {
        OutputParam,
        OutputMoment1,
        AuxGrad
    }
}

impl<Tdata,T,TLengths,rowWiseAdagradT> 
RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientApproxOp<Tdata,T,TLengths,rowWiseAdagradT> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws),
            epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5)),
            weight_decay_( this->template GetSingleArgument<float>("weight_decay", 0.f)) 
        VLOG(1)
            << "gradient optimization operator in use: "
            << "RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientApproxOp";
        const T decay = this->template GetSingleArgument<T>("decay", 1.0);
        CAFFE_ENFORCE_EQ(
            decay, 1.0, "Decay is not supported for SparseSimdAdagradOp");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // Enforce shapes
        CAFFE_ENFORCE_EQ(Input(PARAM).sizes()[0], Input(MOMENT_1).numel());
        CAFFE_ENFORCE_EQ(Input(LR).numel(), 1);
        CAFFE_ENFORCE_EQ(
            Input(PARAM).size_from_dim(1),
            Input(GRAD).size_from_dim(Input(INDICES).dim()));

        return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, Input(INDICES));
        */
    }
    
    #[inline] pub fn do_run_with_type<SIndex>(&mut self) -> bool {
        todo!();
        /*
            if (weight_decay_ == 0.0f) {
          return DoRunWithType<SIndex, false>();
        } else {
          return DoRunWithType<SIndex, true>();
        }
        */
    }
    
    #[inline] fn do_run_with_type_impl<SIndex, const HAS_WEIGHT_DECAY: bool>(&mut self) -> bool {
        todo!();
        /*
            const auto* lr = Input(LR).template data<T>();
        Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
        Output(OUTPUT_MOMENT_1)->ResizeLike(Input(MOMENT_1));

        auto& segmentGradsInput = Input(GRAD);
        auto& lengthsInput = Input(LENGTHS);

        CAFFE_ENFORCE_EQ(lengthsInput.dim(), 1, "LENGTHS must be a vector");
        auto numSegments = lengthsInput.size(0);
        CAFFE_ENFORCE_GT(segmentGradsInput.dim(), 0);
        CAFFE_ENFORCE_EQ(numSegments, segmentGradsInput.size(0));
        const auto* lengths = lengthsInput.template data<TLengths>();

        auto n = Input(INDICES).numel();

        const auto* indices = Input(INDICES).template data<SIndex>();
        const auto* gradIn = segmentGradsInput.template data<T>();
        const auto* paramIn = Input(PARAM).template data<Tdata>();
        const auto* momentIn = Input(MOMENT_1).template data<T>();
        const auto* auxParamIn = Input(AUX_PARAM).template data<T>();

        auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<Tdata>();
        auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<T>();
        Output(AUX_GRAD)->Resize(n);
        auto* auxGrad = Output(AUX_GRAD)->template mutable_data<T>();

        CAFFE_ENFORCE_EQ(
            paramIn, paramOut, "RowWiseSparseAdagrad must use inplace param");
        CAFFE_ENFORCE_EQ(
            momentIn, momentOut, "RowWiseSparseAdagrad must use inplace momentum");

        if (numSegments == 0) {
          return true;
        }

        auto block_size = segmentGradsInput.size_from_dim(1);

        // Enforce:
        // Input(embedding/momentum) == outputs(embedding/momentum)
        CAFFE_ENFORCE_EQ(
            Input(PARAM).numel() / block_size,
            Input(MOMENT_1).numel(),
            "Input Param size: ",
            Input(PARAM).numel(),
            " Block size: ",
            block_size,
            " Input Moment size: ",
            Input(MOMENT_1).numel());

        std::vector<T> temp_grad(block_size);
        int dataIndex = 0;
        for (auto rangeIndex = 0; rangeIndex < numSegments; ++rangeIndex) {
          auto offsetI = rangeIndex * block_size;
          const float* g = gradIn + offsetI;

          float g_sq_avg;
          if (block_size > 1 && !HAS_WEIGHT_DECAY) {
            g_sq_avg = internal::compute_square_average_inlined_(g, block_size);
          }

          for (auto start = dataIndex; dataIndex < start + lengths[rangeIndex];
               ++dataIndex) {
            std::size_t idx = indices[dataIndex];
            auto offsetIdx = idx * block_size;
            auto localOffset = dataIndex - start;

            // Enforce:
            // access within range
            // gradient access within range
            CAFFE_ENFORCE_GE(
                Input(PARAM).numel(),
                block_size + offsetIdx,
                this->debug_def().input(PARAM),
                ", out of bound,  idx:",
                idx,
                " for input dataIndex:",
                dataIndex,
                " and block size:",
                block_size,
                " max size:",
                Input(PARAM).numel());

            int i = 0;
            float acc = 0.0f;

    #ifdef __AVX__
            constexpr int VLEN = 8;
            __m256 acc_v = _mm256_setzero_ps();
            __m256 scalar_v = _mm256_set1_ps(auxParamIn[localOffset]);

            if (std::is_same<Tdata, float>::value) {
              for (; i < block_size / VLEN * VLEN; i += VLEN) {
                __m256 a_v = _mm256_loadu_ps(g + i);
                __m256 b_v = _mm256_loadu_ps(
                    reinterpret_cast<const float*>(paramIn + offsetIdx + i));
                __m256 c_v = _mm256_mul_ps(a_v, b_v);
                acc_v = _mm256_add_ps(acc_v, c_v);
                _mm256_storeu_ps(&temp_grad[i], _mm256_mul_ps(a_v, scalar_v));
              }
            } else if (std::is_same<Tdata, at::Half>::value) {
              for (; i < block_size / VLEN * VLEN; i += VLEN) {
                __m256 a_v = _mm256_loadu_ps(g + i);
                __m256 b_v = _mm256_cvtph_ps(
                    _mm_load_si128((__m128i*)(paramIn + offsetIdx + i)));
                __m256 c_v = _mm256_mul_ps(a_v, b_v);
                acc_v = _mm256_add_ps(acc_v, c_v);
                _mm256_storeu_ps(&temp_grad[i], _mm256_mul_ps(a_v, scalar_v));
              }
            } else {
              CAFFE_THROW("Unsupported type for Embedding");
            }

            alignas(64) float temp[VLEN];
            _mm256_store_ps(temp, acc_v);
            for (int j = 0; j < VLEN; ++j) {
              acc += temp[j];
            }
    #endif

            for (; i < block_size; ++i) {
              float a = g[i];
              acc += a * paramIn[offsetIdx + i];
              temp_grad[i] = a * auxParamIn[localOffset];
            }
            auxGrad[dataIndex] = acc;

            if (block_size == 1) {
              float gi = std::fma(weight_decay_, paramIn[idx], temp_grad[0]);
              float hi = momentOut[idx] = momentIn[idx] + gi * gi;
              paramOut[idx] =
                  paramIn[idx] + lr[0] / (std::sqrt(hi) + epsilon_) * gi;
            } else {
              // prefetching
              const int prefdist_T0 = 16;
              int i_pref = (dataIndex < n - prefdist_T0) ? dataIndex + prefdist_T0
                                                         : dataIndex;
              std::size_t idx_pref = indices[i_pref];

              if (HAS_WEIGHT_DECAY) {
                g_sq_avg =
                    internal::compute_square_average_with_weight_decay_inlined_(
                        temp_grad.data(),
                        paramOut + offsetIdx,
                        block_size,
                        weight_decay_);
              }

              kernel_(
                  block_size,

                  paramOut + offsetIdx,
                  &paramOut[idx_pref * block_size],

                  temp_grad.data(),
                  g_sq_avg *
                      (HAS_WEIGHT_DECAY
                           ? 1
                           : auxParamIn[localOffset] * auxParamIn[localOffset]),

                  momentOut + idx,
                  momentOut + idx_pref,

                  epsilon_,
                  lr[0],
                  HAS_WEIGHT_DECAY ? weight_decay_ : 0.0f);
            }
          }
        }
        CAFFE_ENFORCE_EQ(dataIndex, n);

        return true;
        */
    }
}

struct RowWiseAdagradUpdateInlined {

}

/**
  | w_n - prefetch ptr
  | 
  | h_n - prefetch ptr
  |
  */
impl RowWiseAdagradUpdateInlined {
    
    #[inline] pub fn invoke(&mut self, 
        n:            i32,
        w:            *mut f32,
        w_n:          *mut f32,
        g:            *const f32,
        g_sq_avg:     f32,
        h:            *mut f32,
        h_n:          *mut f32,
        epsilon:      f32,
        lr:           f32,
        weight_decay: f32)  {
        
        todo!();
        /*
            #ifdef __AVX__
        constexpr int kSize = 8;
        _mm_prefetch(reinterpret_cast<const char*>(h_n), _MM_HINT_T0);
    #endif
        float hi = *h = *h + g_sq_avg;
        float float_step = lr / (std::sqrt(hi) + epsilon);

        int i = 0;

    #ifdef __AVX__
        __m256 step = _mm256_set1_ps(float_step);
        __m256 weight_decay_v = _mm256_set1_ps(weight_decay);

        for (i = 0; i + kSize <= N; i += kSize) {
          _mm_prefetch(reinterpret_cast<const char*>(&w_n[i]), _MM_HINT_T0);

          __m256 gi = _mm256_loadu_ps(g + i);
          __m256 wi = _mm256_loadu_ps(w + i);
          if (weight_decay != 0.0f) {
    #ifdef __FMA__
            gi = _mm256_fmadd_ps(weight_decay_v, wi, gi);
    #else
            gi = _mm256_add_ps(_mm256_mul_ps(weight_decay_v, wi), gi);
    #endif
          }

          _mm256_storeu_ps(w + i, _mm256_add_ps(wi, _mm256_mul_ps(gi, step)));
        }
    #endif

        for (; i < N; ++i) {
          float gi =
              weight_decay != 0.0f ? std::fma(weight_decay, w[i], g[i]) : g[i];
          w[i] = w[i] + gi * float_step;
        }
        */
    }
}

/**
  | Match the GPU Approx op, here Approx and Exact
  | are the same for
  | RowWiseSparseAdagradFusedWithSparseLengthsSumGradient
  | op
  */

/**
  | Fused operator of
  | 
  | SparseLengthsIndicesInGradientSumGradient
  | (gradient of SparseLengthsSum) +
  | 
  | RowWiseSparseAdagrad.
  | 
  | Given inputs (param, moment, indices,
  | grad, lr), runs the row-wise sparse
  | 
  | AdaGrad update on (param, grad, moment[indices],
  | lr), and returns (new_param, new_moment)
  | as in the dense case. Additional input
  | (lengths) is for fused
  | 
  | SparseLengthsSumGradient operator.
  |
  */
register_cpu_operator!{
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradientApprox,
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp<
        float,
        float,
        int,
        rowwise_adagrad_update_inlined,
        /*is_mean=*/false>
}

register_cpu_operator_with_engine!{
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradientApprox,
    SIMD,
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp::<
        f32,
        f32,
        i32,
        rowwise_adagrad_update_inlined,
        IsMeanFlag>
}

num_inputs!{RowWiseSparseAdagradFusedWithSparseLengthsSumGradientApprox, 6}

num_outputs!{RowWiseSparseAdagradFusedWithSparseLengthsSumGradientApprox, 2}

inputs!{RowWiseSparseAdagradFusedWithSparseLengthsSumGradientApprox, 
    0 => ("param",    "Parameters to be updated"),
    1 => ("moment",   "Moment history"),
    2 => ("indices",  "Integer vector containing indices of the first dimension of param for the slices that are being updated"),
    3 => ("grad",     "Gradient computed"),
    4 => ("lr",       "learning rate"),
    5 => ("lengths",  "Non negative vector with sum of elements equal to indices length")
}

outputs!{RowWiseSparseAdagradFusedWithSparseLengthsSumGradientApprox, 
    0 => ("output_param",  "Updated parameters"),
    1 => ("output_moment", "Updated moment")
}

args!{RowWiseSparseAdagradFusedWithSparseLengthsSumGradientApprox, 
    0 => ("round_option", "rounding option: 0 for nearest rounding, 1 for stochastic rounding"),
    1 => ("epsilon",      "Default 1e-5")
}

enforce_one_to_one_inplace!{RowWiseSparseAdagradFusedWithSparseLengthsSumGradientApprox}

/**
  | Fused operator of
  | 
  | SparseLengthsIndicesInGradientMeanGradient
  | (gradient of SparseLengthsMean) +
  | 
  | RowWiseSparseAdagrad.
  | 
  | Given inputs (param, moment, indices,
  | grad, lr), runs the row-wise sparse
  | 
  | AdaGrad update on (param, grad, moment[indices],
  | lr), and returns (new_param, new_moment)
  | as in the dense case. Additional input
  | (lengths) is for fused
  | 
  | SparseLengthsMeanGradient operator.
  |
  */
register_cpu_operator!{
    RowWiseSparseAdagradFusedWithSparseLengthsMeanGradient,
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp<
        float,
        float,
        int,
        rowwise_adagrad_update_inlined,
        /*is_mean=*/true>
}

register_cpu_operator_with_engine!{
    RowWiseSparseAdagradFusedWithSparseLengthsMeanGradient,
    SIMD,
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp::<
        f32,
        f32,
        i32,
        rowwise_adagrad_update_inlined,
        IsMeanFlag>
}

num_inputs!{RowWiseSparseAdagradFusedWithSparseLengthsMeanGradient, 6}

num_outputs!{RowWiseSparseAdagradFusedWithSparseLengthsMeanGradient, 2}

inputs!{RowWiseSparseAdagradFusedWithSparseLengthsMeanGradient, 
    0 => ("param",          "Parameters to be updated"),
    1 => ("moment",         "Moment history"),
    2 => ("indices",        "Integer vector containing indices of the first dimension of param for the slices that are being updated"),
    3 => ("grad",           "Gradient computed"),
    4 => ("lr",             "learning rate"),
    5 => ("lengths",        "Non negative vector with sum of elements equal to indices length")
}

outputs!{RowWiseSparseAdagradFusedWithSparseLengthsMeanGradient, 
    0 => ("output_param",   "Updated parameters"),
    1 => ("output_moment",  "Updated moment")
}

args!{RowWiseSparseAdagradFusedWithSparseLengthsMeanGradient, 
    0 => ("epsilon",        "Default 1e-5")
}

enforce_one_to_one_inplace!{RowWiseSparseAdagradFusedWithSparseLengthsMeanGradient}

/**
  | Match the GPU Approx op, here Approx and Exact
  | are the same for
  | RowWiseSparseAdagradFusedWithSparseLengthsMeanGradient
  | op
  */

/**
  | Fused operator of
  | 
  | SparseLengthsIndicesInGradientMeanGradient
  | (gradient of SparseLengthsMean) +
  | 
  | RowWiseSparseAdagrad.
  | 
  | Given inputs (param, moment, indices,
  | grad, lr), runs the row-wise sparse
  | 
  | AdaGrad update on (param, grad, moment[indices],
  | lr), and returns (new_param, new_moment)
  | as in the dense case. Additional input
  | (lengths) is for fused
  | 
  | SparseLengthsMeanGradient operator.
  |
  */
register_cpu_operator!{
    RowWiseSparseAdagradFusedWithSparseLengthsMeanGradientApprox,
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp::<
        f32,
        f32,
        i32,
        rowwise_adagrad_update_inlined,
        IsMeanFlag>
}

pub const IsMeanFlag: bool = true;

register_cpu_operator_with_engine!{
    RowWiseSparseAdagradFusedWithSparseLengthsMeanGradientApprox,
    SIMD,
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp::<
        f32,
        f32,
        i32,
        rowwise_adagrad_update_inlined,
        IsMeanFlag>
}

num_inputs!{RowWiseSparseAdagradFusedWithSparseLengthsMeanGradientApprox, 6}

num_outputs!{RowWiseSparseAdagradFusedWithSparseLengthsMeanGradientApprox, 2}

inputs!{RowWiseSparseAdagradFusedWithSparseLengthsMeanGradientApprox, 
    0 => ("param",         "Parameters to be updated"),
    1 => ("moment",        "Moment history"),
    2 => ("indices",       "Integer vector containing indices of the first dimension of param for the slices that are being updated"),
    3 => ("grad",          "Gradient computed"),
    4 => ("lr",            "learning rate"),
    5 => ("lengths",       "Non negative vector with sum of elements equal to indices length")
}

outputs!{RowWiseSparseAdagradFusedWithSparseLengthsMeanGradientApprox, 
    0 => ("output_param",  "Updated parameters"),
    1 => ("output_moment", "Updated moment")
}

args!{RowWiseSparseAdagradFusedWithSparseLengthsMeanGradientApprox, 
    0 => ("round_option", "rounding option: 0 for nearest rounding, 1 for stochastic rounding"),
    1 => ("epsilon",      "Default 1e-5")
}

enforce_one_to_one_inplace!{RowWiseSparseAdagradFusedWithSparseLengthsMeanGradientApprox}

/**
  | Fused operator of SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient
  | (gradient of SparseLengthsWeightedSum)
  | + RowWiseSparseAdagrad, where weights
  | are positional weights computed with
  | LengthsRangeFill + Gather pattern.
  | 
  | Given inputs (param, moment, indices,
  | grad, lr), runs the row-wise sparse
  | 
  | AdaGrad update on (param, grad, moment[indices],
  | lr), and returns (new_param, new_moment)
  | as in the dense case.
  | 
  | There're auxiliary inputs (aux_param)
  | for which gradient is computed and returns
  | (aux_grad).
  | 
  | Yet additional input (lengths) is for
  | fused SparseLengthsWeightedSumGradient
  | operator.
  |
  */
register_cpu_operator!{
    RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradient,
    RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientOp::<
        f32,
        f32,
        i32,
        rowwise_adagrad_update_inlined>
}

register_cpu_operator_with_engine!{
    RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradient,
    SIMD,
    RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientOp::<
        f32,
        f32,
        i32,
        rowwise_adagrad_update_inlined>
}

num_inputs!{RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradient, 7}

num_outputs!{RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradient, 3}

inputs!{RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradient, 
    0 => ("param",       "Parameters to be updated"),
    1 => ("moment",      "Moment history"),
    2 => ("aux_param",   "Auxiliary parameters to be updated"),
    3 => ("indices",     "Integer vector containing indices of the first dimension of param for the slices that are being updated"),
    4 => ("grad",        "Gradient computed"),
    5 => ("lr",          "learning rate"),
    6 => ("lengths",     "Non negative vector with sum of elements equal to indices length")
}

outputs!{RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradient, 
    0 => ("output_param",  "Updated parameters"),
    1 => ("output_moment", "Updated moment"),
    2 => ("aux_grad",      "Auxiliary gradient")
}

args!{RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradient, 
    0 => ("epsilon", "Default 1e-5")
}

enforce_inplace!{RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradient, vec![(0, 0), (1, 1)]}

/**
  | Approximately fused operator of
  | 
  | SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient
  | (gradient of SparseLengthsWeightedSum)
  | + RowWiseSparseAdagrad, where weights
  | are positional weights computed with
  | LengthsRangeFill + Gather pattern.
  | 
  | Given inputs (param, moment, indices,
  | grad, lr), runs the row-wise sparse
  | 
  | AdaGrad update on (param, grad, moment[indices],
  | lr), and returns (new_param, new_moment)
  | as in the dense case.
  | 
  | There's race condition w.r.t. ordering
  | between reading params and writing
  | to param, hence the name Approx.
  | 
  | There're auxiliary inputs (aux_param)
  | for which gradient is computed and returns
  | (aux_grad).
  | 
  | Yet additional input (lengths) is for
  | fused SparseLengthsWeightedSumGradient
  | operator.
  |
  */
register_cpu_operator!{
    RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientApprox,
    RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientApproxOp::<
        f32,
        f32,
        i32,
        rowwise_adagrad_update_inlined>
}

register_cpu_operator_with_engine!{
    RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientApprox,
    SIMD,
    RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientApproxOp::<
        f32,
        f32,
        i32,
        rowwise_adagrad_update_inlined>
}

num_inputs!{RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientApprox, 7}

num_outputs!{RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientApprox, 3}

inputs!{RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientApprox, 
    0 => ("param",           "Parameters to be updated"),
    1 => ("moment",          "Moment history"),
    2 => ("aux_param",       "Auxiliary parameters to be updated"),
    3 => ("indices",         "Integer vector containing indices of the first dimension of param for the slices that are being updated"),
    4 => ("grad",            "Gradient computed"),
    5 => ("lr",              "learning rate"),
    6 => ("lengths",         "Non negative vector with sum of elements equal to indices length")
}

outputs!{RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientApprox, 
    0 => ("output_param",    "Updated parameters"),
    1 => ("output_moment",   "Updated moment"),
    2 => ("aux_grad",        "Auxiliary gradient")
}

args!{RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientApprox, 
    0 => ("epsilon",         "Default 1e-5")
}

enforce_inplace!{
    RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientApprox, 
    vec![(0, 0), (1, 1)]
}
