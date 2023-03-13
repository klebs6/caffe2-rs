crate::ix!();


/**
  | Fused operator of
  | 
  | SparseLengthsIndicesInGradientSumGradient
  | (gradient of SparseLengthsSum) + SparseAdagrad.
  | 
  | Given inputs (param, moment, indices,
  | grad, lr), runs the sparse AdaGrad update
  | on (param, grad, moment[indices],
  | lr), and returns (new_param, new_moment)
  | as in the dense case. Additional input
  | (lengths) is for fused
  | 
  | SparseLengthsIndicesInGradientSumGradient
  | operator.
  | 
  | typename Tdata, // embedding and momentum
  | types typename T, // everything else
  | bool is_mean = false>
  |
  */
pub struct SparseAdagradFusedWithSparseLengthsSumGradientOp<Tdata,T,TLengths,AdagradT,const is_mean: bool> {
    storage: OperatorStorage,
    context: CPUContext,

    epsilon:       T,
    weight_decay:  T,
    kernel:        AdagradT,
    grad_buffer:   Tensor, // default = CPU
    phantomA: PhantomData<TLengths>,
    phantomB: PhantomData<Tdata>,
}

num_inputs!{SparseAdagradFusedWithSparseLengthsSumGradient, 6}

num_outputs!{SparseAdagradFusedWithSparseLengthsSumGradient, 2}

inputs!{SparseAdagradFusedWithSparseLengthsSumGradient, 
    0 => ("param",            "Parameters to be updated"),
    1 => ("moment",           "Moment history"),
    2 => ("indices",          "Integer vector containing indices of the first dimension of param for the slices that are being updated"),
    3 => ("grad",             "Gradient computed"),
    4 => ("lr",               "learning rate"),
    5 => ("lengths",          "Non negative vector with sum of elements equal to indices length")
}

outputs!{SparseAdagradFusedWithSparseLengthsSumGradient, 
    0 => ("output_param",     "Updated parameters"),
    1 => ("output_moment",    "Updated moment")
}

args!{SparseAdagradFusedWithSparseLengthsSumGradient, 
    0 => ("epsilon",          "Default 1e-5")
}

enforce_one_to_one_inplace!{SparseAdagradFusedWithSparseLengthsSumGradient}

register_cpu_operator!{
    SparseAdagradFusedWithSparseLengthsSumGradient,
    SparseAdagradFusedWithSparseLengthsSumGradientOp<
        float,
        float,
        int,
        adagrad_update_prefetch_inlined,
        /*is_mean=*/false>
}

input_tags!{
    SparseAdagradFusedWithSparseLengthsSumGradientOp
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
    SparseAdagradFusedWithSparseLengthsSumGradientOp
    {
        OutputParam,
        OutputMoment1
    }
}

impl< Tdata, T, TLengths, AdagradT, const is_mean: bool> 
SparseAdagradFusedWithSparseLengthsSumGradientOp< Tdata, T, TLengths, AdagradT, is_mean> 
{
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws),
            epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5)),
            weight_decay_( this->template GetSingleArgument<float>("weight_decay", 0.f)) 

        VLOG(1) << "gradient optimization operator in use: "
                << "SparseAdagradFusedWithSparseLengthsSumGradientOp"
                << " weight_decay_=" << weight_decay_;
        const T decay = this->template GetSingleArgument<T>("decay", 1.0);
        CAFFE_ENFORCE_EQ(
            decay, 1.0, "Decay is not supported for SparseSimdAdagradOp");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
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

        const auto* indices = Input(INDICES).template data<SIndex>();
        const auto* gradIn = segmentGradsInput.template data<T>();
        const auto* paramIn = Input(PARAM).template data<Tdata>();
        const auto* momentIn = Input(MOMENT_1).template data<Tdata>();
        auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<Tdata>();
        auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<Tdata>();

        if (numSegments == 0) {
          return true;
        }
        auto block_size = segmentGradsInput.size_from_dim(1);

        // Enforce:
        // input(embedding/momentum) == outputs(embedding/momentum)
        CAFFE_ENFORCE_EQ(
            Input(PARAM).numel(),
            Input(MOMENT_1).numel(),
            "Input Param size: ",
            Input(PARAM).numel(),
            " Input Moment size: ",
            Input(MOMENT_1).numel());

        int dataIndex = 0;
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

            if (block_size == 1) {
              float gi = std::fma(
                  weight_decay_,
                  paramIn[idx],
                  is_mean ? grad_buffer_data[offsetI] : gradIn[offsetI]);
              float hi = momentOut[idx] = momentIn[idx] + gi * gi;
              paramOut[idx] =
                  paramIn[idx] + lr[0] * gi / (std::sqrt(hi) + epsilon_);
            } else {
              // prefetching
              const int prefdist_T0 = 16;
              int i_pref = (dataIndex < n - prefdist_T0) ? dataIndex + prefdist_T0
                                                         : dataIndex;
              std::size_t idx_pref = indices[i_pref];
              kernel_(
                  block_size,

                  paramIn + offsetIdx,
                  &paramIn[idx_pref * block_size],

                  is_mean ? grad_buffer_data + offsetI : gradIn + offsetI,

                  momentIn + offsetIdx,
                  &momentIn[idx_pref * block_size],

                  paramOut + offsetIdx,
                  &paramOut[idx_pref * block_size],

                  momentOut + offsetIdx,
                  &momentOut[idx_pref * block_size],

                  epsilon_,
                  lr[0],
                  weight_decay_);
            }
          }
        }
        CAFFE_ENFORCE_EQ(dataIndex, n);

        return true;
        */
    }
}

///--------------------------------------------
pub struct SparseAdagradFusedWithSparseLengthsWeightedSumGradientOp<Tdata,T,TLengths,AdagradT> {

    storage: OperatorStorage,
    context: CPUContext,

    epsilon:       T,
    weight_decay:  T,
    kernel:        AdagradT,
    phantomA: PhantomData<TLengths>,
    phantomB: PhantomData<Tdata>,
}

input_tags!{
    SparseAdagradFusedWithSparseLengthsWeightedSumGradientOp
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
    SparseAdagradFusedWithSparseLengthsWeightedSumGradientOp
    {
        OutputParam,
        OutputMoment1,
        AuxGrad
    }
}


impl<Tdata,T,TLengths,AdagradT> 
SparseAdagradFusedWithSparseLengthsWeightedSumGradientOp<Tdata,T,TLengths,AdagradT> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws),
            epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5)),
            weight_decay_( this->template GetSingleArgument<float>("weight_decay", 0.f)) 

        VLOG(1) << "gradient optimization operator in use: "
                << "SparseAdagradFusedWithSparseLengthsWeightedSumGradientOp";
        const T decay = this->template GetSingleArgument<T>("decay", 1.0);
        CAFFE_ENFORCE_EQ(
            decay, 1.0, "Decay is not supported for SparseSimdAdagradOp");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
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

        const auto* indices = Input(INDICES).template data<SIndex>();
        const auto* gradIn = segmentGradsInput.template data<T>();
        const auto* paramIn = Input(PARAM).template data<Tdata>();
        const auto* momentIn = Input(MOMENT_1).template data<Tdata>();
        const auto* auxParamIn = Input(AUX_PARAM).template data<T>();

        auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<Tdata>();
        auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<Tdata>();
        Output(AUX_GRAD)->Resize(n);
        auto* auxGrad = Output(AUX_GRAD)->template mutable_data<T>();

        if (numSegments == 0) {
          return true;
        }

        auto block_size = segmentGradsInput.size_from_dim(1);

        // Enforce:
        // input(embedding/momentum) == outputs(embedding/momentum)
        CAFFE_ENFORCE_EQ(
            Input(PARAM).numel(),
            Input(MOMENT_1).numel(),
            "Input Param size: ",
            Input(PARAM).numel(),
            " Input Moment size: ",
            Input(MOMENT_1).numel());

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

            internal::dot<T, Tdata, T>(
                block_size,
                gradIn + offsetI,
                paramIn + offsetIdx,
                auxGrad + dataIndex,
                &context_);
          }
        }
        CAFFE_ENFORCE_EQ(dataIndex, n);

        dataIndex = 0;
        for (auto rangeIndex = 0; rangeIndex < numSegments; ++rangeIndex) {
          for (auto start = dataIndex; dataIndex < start + lengths[rangeIndex];
               ++dataIndex) {
            std::size_t idx = indices[dataIndex];
            auto offsetI = rangeIndex * block_size;
            auto offsetIdx = idx * block_size;
            auto localOffset = dataIndex - start;

            for (int i = 0; i < block_size; ++i) {
              temp_grad[i] = auxParamIn[localOffset] * gradIn[offsetI + i];
            }

            if (block_size == 1) {
              float gi = std::fma(weight_decay_, paramIn[idx], temp_grad[0]);
              float hi = momentOut[idx] = momentIn[idx] + gi * gi;
              paramOut[idx] =
                  paramIn[idx] + lr[0] * gi / (std::sqrt(hi) + epsilon_);
            } else {
              // prefetching
              const int prefdist_T0 = 16;
              int i_pref = (dataIndex < n - prefdist_T0) ? dataIndex + prefdist_T0
                                                         : dataIndex;
              std::size_t idx_pref = indices[i_pref];
              kernel_(
                  block_size,

                  paramIn + offsetIdx,
                  &paramIn[idx_pref * block_size],

                  temp_grad.data(),

                  momentIn + offsetIdx,
                  &momentIn[idx_pref * block_size],

                  paramOut + offsetIdx,
                  &paramOut[idx_pref * block_size],

                  momentOut + offsetIdx,
                  &momentOut[idx_pref * block_size],

                  epsilon_,
                  lr[0],
                  weight_decay_);
            }
          }
        }

        return true;
        */
    }
}

/**
  ------------------------------------
    typename Tdata, // embedding and momentum types
    typename T, // everything else
*/
pub struct SparseAdagradFusedWithSparseLengthsWeightedSumGradientApproxOp<Tdata,T,TLengths,AdagradT> {
    storage:       OperatorStorage,
    context:       CPUContext,
    epsilon:       T,
    weight_decay:  T,
    kernel:        AdagradT,
    phantomA: PhantomData<TLengths>,
    phantomB: PhantomData<Tdata>,
}

impl<Tdata,T,TLengths,AdagradT> Operator for 
SparseAdagradFusedWithSparseLengthsWeightedSumGradientApproxOp<Tdata,T,TLengths,AdagradT> 
{


}

input_tags!{
    SparseAdagradFusedWithSparseLengthsWeightedSumGradientApproxOp
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
    SparseAdagradFusedWithSparseLengthsWeightedSumGradientApproxOp
    {
        OutputParam,
        OutputMoment1,
        AuxGrad
    }
}

impl<Tdata,T,TLengths,AdagradT> 
SparseAdagradFusedWithSparseLengthsWeightedSumGradientApproxOp<Tdata,T,TLengths,AdagradT> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws),
            epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5)),
            weight_decay_( this->template GetSingleArgument<float>("weight_decay", 0.f)) 

        VLOG(1) << "gradient optimization operator in use: "
                << "SparseAdagradFusedWithSparseLengthsWeightedSumGradientApproxOp";
        const T decay = this->template GetSingleArgument<T>("decay", 1.0);
        CAFFE_ENFORCE_EQ(
            decay, 1.0, "Decay is not supported for SparseSimdAdagradOp");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
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

        const auto* indices = Input(INDICES).template data<SIndex>();
        const auto* gradIn = segmentGradsInput.template data<T>();
        const auto* paramIn = Input(PARAM).template data<Tdata>();
        const auto* momentIn = Input(MOMENT_1).template data<Tdata>();
        const auto* auxParamIn = Input(AUX_PARAM).template data<T>();

        auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<Tdata>();
        auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<Tdata>();
        Output(AUX_GRAD)->Resize(n);
        auto* auxGrad = Output(AUX_GRAD)->template mutable_data<T>();

        if (numSegments == 0) {
          return true;
        }

        auto block_size = segmentGradsInput.size_from_dim(1);

        // Enforce:
        // input(embedding/momentum) == outputs(embedding/momentum)
        CAFFE_ENFORCE_EQ(
            Input(PARAM).numel(),
            Input(MOMENT_1).numel(),
            "Input Param size: ",
            Input(PARAM).numel(),
            " Input Moment size: ",
            Input(MOMENT_1).numel());

        std::vector<T> temp_grad(block_size);
        int dataIndex = 0;
        for (auto rangeIndex = 0; rangeIndex < numSegments; ++rangeIndex) {
          for (auto start = dataIndex; dataIndex < start + lengths[rangeIndex];
               ++dataIndex) {
            std::size_t idx = indices[dataIndex];
            auto offsetI = rangeIndex * block_size;
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

            internal::dot<T, Tdata, T>(
                block_size,
                gradIn + offsetI,
                paramIn + offsetIdx,
                auxGrad + dataIndex,
                &context_);

            for (int i = 0; i < block_size; ++i) {
              temp_grad[i] = auxParamIn[localOffset] * gradIn[offsetI + i];
            }

            if (block_size == 1) {
              float gi = std::fma(weight_decay_, paramIn[idx], temp_grad[0]);
              float hi = momentOut[idx] = momentIn[idx] + gi * gi;
              paramOut[idx] =
                  paramIn[idx] + lr[0] * gi / (std::sqrt(hi) + epsilon_);
            } else {
              // prefetching
              const int prefdist_T0 = 16;
              int i_pref = (dataIndex < n - prefdist_T0) ? dataIndex + prefdist_T0
                                                         : dataIndex;
              std::size_t idx_pref = indices[i_pref];
              kernel_(
                  block_size,

                  paramIn + offsetIdx,
                  &paramIn[idx_pref * block_size],

                  temp_grad.data(),

                  momentIn + offsetIdx,
                  &momentIn[idx_pref * block_size],

                  paramOut + offsetIdx,
                  &paramOut[idx_pref * block_size],

                  momentOut + offsetIdx,
                  &momentOut[idx_pref * block_size],

                  epsilon_,
                  lr[0],
                  weight_decay_);
            }
          }
        }
        CAFFE_ENFORCE_EQ(dataIndex, n);

        return true;
        */
    }
}

pub struct adagrad_update_prefetch_inlined { }

impl adagrad_update_prefetch_inlined {

    /**
     | const float* w_n, // prefetch ptr
     | const float* h_n, // prefetch ptr
     | float* nw_n, // prefetch ptr
     | float* nh_n, // prefetch ptr
     */
    #[inline] pub fn invoke(&mut self, 
        n:            i32,
        w:            *const f32,
        w_n:          *const f32,
        g:            *const f32,
        h:            *const f32,
        h_n:          *const f32,
        nw:           *mut f32,
        nw_n:         *mut f32,
        nh:           *mut f32,
        nh_n:         *mut f32,
        epsilon:      f32,
        lr:           f32,
        weight_decay: f32)  {

        todo!();
        /*
            return internal::adagrad_update_prefetch_inlined(
            N, w, w_n, g, h, h_n, nw, nw_n, nh, nh_n, epsilon, lr, weight_decay);
        */
    }
}

/**
  | Fused operator of
  | SparseLengthsIndicesInGradientSumGradient
  | (gradient of SparseLengthsSum) + SparseAdagrad.
  | 
  | Given inputs (param, moment, indices,
  | grad, lr), runs the sparse AdaGrad update
  | on (param, grad, moment[indices],
  | lr), and returns (new_param, new_moment)
  | as in the dense case. Additional input
  | (lengths) is for fused
  | 
  | SparseLengthsIndicesInGradientSumGradient
  | operator.
  |
  */
register_cpu_operator!{
    SparseAdagradFusedWithSparseLengthsSumGradientApprox,
    SparseAdagradFusedWithSparseLengthsSumGradientOp<
    float,
        float,
        int,
        adagrad_update_prefetch_inlined,
        /*is_mean=*/false>
}

/**
  | Match the GPU Approx op, here Approx and Exact
  | are the same for
  | SparseAdagradFusedWithSparseLengthsSumGradient
  | op
  */
num_inputs!{SparseAdagradFusedWithSparseLengthsSumGradientApprox, 6}

num_outputs!{SparseAdagradFusedWithSparseLengthsSumGradientApprox, 2}

inputs!{SparseAdagradFusedWithSparseLengthsSumGradientApprox, 
    0 => ("param",           "Parameters to be updated"),
    1 => ("moment",          "Moment history"),
    2 => ("indices",         "Integer vector containing indices of the first dimension of param for the slices that are being updated"),
    3 => ("grad",            "Gradient computed"),
    4 => ("lr",              "learning rate"),
    5 => ("lengths",         "Non negative vector with sum of elements equal to indices length")
}

outputs!{SparseAdagradFusedWithSparseLengthsSumGradientApprox, 
    0 => ("output_param",    "Updated parameters"),
    1 => ("output_moment",   "Updated moment")
}

args!{SparseAdagradFusedWithSparseLengthsSumGradientApprox, 
    0 => ("epsilon",         "Default 1e-5")
}

enforce_one_to_one_inplace!{SparseAdagradFusedWithSparseLengthsSumGradientApprox}

/**
  | Fused operator of
  | SparseLengthsIndicesInGradientMeanGradient
  | (gradient of SparseLengthsMean) +
  | SparseAdagrad.
  | 
  | Given inputs (param, moment, indices,
  | grad, lr), runs the sparse AdaGrad update
  | on (param, grad, moment[indices],
  | lr), and returns (new_param, new_moment)
  | as in the dense case. Additional input
  | (lengths) is for fused
  | 
  | SparseLengthsIndicesInGradientMeanGradient
  | operator.
  |
  */
register_cpu_operator!{
    SparseAdagradFusedWithSparseLengthsMeanGradient,
    SparseAdagradFusedWithSparseLengthsSumGradientOp<
        float,
        float,
        int,
        adagrad_update_prefetch_inlined,
        /*is_mean=*/true>
}

num_inputs!{SparseAdagradFusedWithSparseLengthsMeanGradient, 6}

num_outputs!{SparseAdagradFusedWithSparseLengthsMeanGradient, 2}

inputs!{
    SparseAdagradFusedWithSparseLengthsMeanGradient, 
    0 => ("param",                "Parameters to be updated"),
    1 => ("moment",               "Moment history"),
    2 => ("indices",              "Integer vector containing indices of the first dimension of param for the slices that are being updated"),
    3 => ("grad",                 "Gradient computed"),
    4 => ("lr",                   "learning rate"),
    5 => ("lengths",              "Non negative vector with sum of elements equal to indices length")
}

outputs!{
    SparseAdagradFusedWithSparseLengthsMeanGradient, 
    0 => ("output_param",         "Updated parameters"),
    1 => ("output_moment",        "Updated moment")
}

args!{SparseAdagradFusedWithSparseLengthsMeanGradient, 
    0 => ("epsilon",              "Default 1e-5")
}

enforce_one_to_one_inplace!{SparseAdagradFusedWithSparseLengthsMeanGradient}

/**
  | Fused operator of
  | SparseLengthsIndicesInGradientMeanGradient
  | (gradient of SparseLengthsMean) +
  | SparseAdagrad.
  | 
  | Given inputs (param, moment, indices,
  | grad, lr), runs the sparse AdaGrad update
  | on (param, grad, moment[indices],
  | lr), and returns (new_param, new_moment)
  | as in the dense case. Additional input
  | (lengths) is for fused
  | 
  | SparseLengthsIndicesInGradientMeanGradient
  | operator.
  |
  */
register_cpu_operator!{SparseAdagradFusedWithSparseLengthsMeanGradientApprox,
    SparseAdagradFusedWithSparseLengthsSumGradientOp<
    float,
    float,
    int,
    adagrad_update_prefetch_inlined,
    /*is_mean=*/true>}

/**
  | Match the GPU Approx op, here Approx and Exact
  | are the same for
  | SparseAdagradFusedWithSparseLengthsMeanGradient
  | op
  */
num_inputs!{SparseAdagradFusedWithSparseLengthsMeanGradientApprox, 6}

num_outputs!{SparseAdagradFusedWithSparseLengthsMeanGradientApprox, 2}

inputs!{SparseAdagradFusedWithSparseLengthsMeanGradientApprox, 
    0 => ("param",            "Parameters to be updated"),
    1 => ("moment",           "Moment history"),
    2 => ("indices",          "Integer vector containing indices of the first dimension of param for the slices that are being updated"),
    3 => ("grad",             "Gradient computed"),
    4 => ("lr",               "learning rate"),
    5 => ("lengths",          "Non negative vector with sum of elements equal to indices length")
}

outputs!{SparseAdagradFusedWithSparseLengthsMeanGradientApprox, 
    0 => ("output_param",     "Updated parameters"),
    1 => ("output_moment",    "Updated moment")
}

args!{SparseAdagradFusedWithSparseLengthsMeanGradientApprox, 
    0 => ("epsilon",          "Default 1e-5")
}

enforce_one_to_one_inplace!{SparseAdagradFusedWithSparseLengthsMeanGradientApprox}

/**
  | Fused operator of
  | SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient
  | (gradient of SparseLengthsWeightedSum) + SparseAdagrad, where weights are
  | positional weights computed with LengthsRangeFill + Gather pattern.
  | 
  | Given inputs (param, moment, indices,
  | grad, lr), runs the sparse AdaGrad update
  | on (param, grad, moment[indices],
  | lr), and returns (new_param, new_moment)
  | as in the dense case.
  | 
  | There're auxiliary inputs (aux_param)
  | for which gradient is computed and returns
  | (aux_grad).
  | 
  | Yet additional input (lengths) is for
  | fused
  | 
  | SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient
  | operator.
  |
  */
register_cpu_operator!{
    SparseAdagradFusedWithSparseLengthsWeightedSumGradient,
    SparseAdagradFusedWithSparseLengthsWeightedSumGradientOp<
        f32,
        f32,
        i32,
        adagrad_update_prefetch_inlined>
}

num_inputs!{SparseAdagradFusedWithSparseLengthsWeightedSumGradient, 7}

num_outputs!{SparseAdagradFusedWithSparseLengthsWeightedSumGradient, 3}

inputs!{SparseAdagradFusedWithSparseLengthsWeightedSumGradient, 
    0 => ("param",                "Parameters to be updated"),
    1 => ("moment",               "Moment history"),
    2 => ("aux_param",            "Auxiliary parameters to be updated"),
    3 => ("indices",              "Integer vector containing indices of the first dimension of param for the slices that are being updated"),
    4 => ("grad",                 "Gradient computed"),
    5 => ("lr",                   "learning rate"),
    6 => ("lengths",              "Non negative vector with sum of elements equal to indices length")
}

outputs!{SparseAdagradFusedWithSparseLengthsWeightedSumGradient, 
    0 => ("output_param",         "Updated parameters"),
    1 => ("output_moment",        "Updated moment"),
    2 => ("aux_grad",             "Auxiliary gradient")
}

args!{SparseAdagradFusedWithSparseLengthsWeightedSumGradient, 
    0 => ("epsilon",              "Default 1e-5")
}

enforce_inplace!{SparseAdagradFusedWithSparseLengthsWeightedSumGradient, vec![(0, 0), (1, 1)]}

/**
  | Approximately fused operator of
  | 
  | SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient
  | (gradient of SparseLengthsWeightedSum)
  | + SparseAdagrad, where weights are
  | positional weights computed with LengthsRangeFill
  | + Gather pattern.
  | 
  | Given inputs (param, moment, indices,
  | grad, lr), runs the sparse AdaGrad update
  | on (param, grad, moment[indices],
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
  | fused
  | 
  | SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient
  | operator.
  |
  */
register_cpu_operator!{
    SparseAdagradFusedWithSparseLengthsWeightedSumGradientApprox,
    SparseAdagradFusedWithSparseLengthsWeightedSumGradientApproxOp<
        float,
        float,
        int,
        adagrad_update_prefetch_inlined>
}

num_inputs!{SparseAdagradFusedWithSparseLengthsWeightedSumGradientApprox, 7}

num_outputs!{SparseAdagradFusedWithSparseLengthsWeightedSumGradientApprox, 3}

inputs!{SparseAdagradFusedWithSparseLengthsWeightedSumGradientApprox, 
    0 => ("param",                "Parameters to be updated"),
    1 => ("moment",               "Moment history"),
    2 => ("aux_param",            "Auxiliary parameters to be updated"),
    3 => ("indices",              "Integer vector containing indices of the first dimension of param for the slices that are being updated"),
    4 => ("grad",                 "Gradient computed"),
    5 => ("lr",                   "learning rate"),
    6 => ("lengths",              "Non negative vector with sum of elements equal to indices length")
}

outputs!{SparseAdagradFusedWithSparseLengthsWeightedSumGradientApprox, 
    0 => ("output_param",         "Updated parameters"),
    1 => ("output_moment",        "Updated moment"),
    2 => ("aux_grad",             "Auxiliary gradients")
}

args!{SparseAdagradFusedWithSparseLengthsWeightedSumGradientApprox, 
    0 => ("epsilon",              "Default 1e-5")
}

enforce_inplace!{SparseAdagradFusedWithSparseLengthsWeightedSumGradientApprox, vec![(0, 0), (1, 1)]}
