crate::ix!();

/**
  | Given a tensor, apply clip after gradient
  | is applied; when the param is sparse
  | as indicated by valid indices and grad,
  | in-place is required
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct EnsureClippedOp<T, Context> {
    storage: OperatorStorage,
    context: Context,

    min:     T,
    max:     T,
}

impl<T,Context> EnsureClippedOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
                 min_(std::numeric_limits<T>::lowest()),
                 max_(T::max) 

                     if (HasArgument("min")) {
                         min_ = static_cast<T>(this->template GetSingleArgument<float>("min", 0));
                     }
                 if (HasArgument("max")) {
                     max_ = static_cast<T>(this->template GetSingleArgument<float>("max", 0));
                 }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (InputSize() > INDICES) {
          // spares gradient, selective checking clipping
          CAFFE_ENFORCE_EQ(
              Input(PARAM).size_from_dim(1),
              Input(GRAD).size_from_dim(Input(INDICES).dim()));
          return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
              this, Input(INDICES));
        } else {
          auto& X = Input(PARAM);

          auto* Y = Output(OUTPUT_PARAM, X.sizes(), at::dtype<float>());
          EigenVectorMap<float>(Y->template mutable_data<float>(), Y->numel()) =
              ConstEigenVectorMap<float>(X.template data<float>(), X.numel())
                  .cwiseMax(min_)
                  .cwiseMin(max_);
          return true;
        }
        */
    }
}

input_tags!{
    EnsureClippedOp {
        Param,
        Indices,
        Grad
    }
}

output_tags!{
    EnsureClippedOp {
        OutputParam
    }
}

impl EnsureClippedOp<f32, CPUContext> {

    #[inline] pub fn do_run_with_type<SIndex>(&mut self) -> bool {
        todo!();
        /*
            Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
          const auto* indices = Input(INDICES).template data<SIndex>();
          const auto* paramIn = Input(PARAM).template data<float>();
          auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<float>();
          CAFFE_ENFORCE_EQ(paramIn, paramOut);
          // n: number of sparse embeddings to be normalized
          auto n = Input(INDICES).numel();
          if (n == 0) {
            return true;
          }
          // embedding length, e.g. 32, 64, 128
          auto block_size = Input(GRAD).numel() / n;
          for (int i = 0; i < n; ++i) {
            auto idx = indices[i];
            auto offsetIdx = idx * block_size;
            EigenVectorMap<float>(paramOut + offsetIdx, block_size) =
                ConstEigenVectorMap<float>(paramIn + offsetIdx, block_size)
                    .cwiseMax(min_)
                    .cwiseMin(max_);
          }
          return true;
        */
    }
}

register_cpu_operator!{
    EnsureClipped, 
    EnsureClippedOp<f32, CPUContext>
}

num_inputs!{EnsureClipped, (1,3)}

num_outputs!{EnsureClipped, 1}

inputs!{EnsureClipped, 
    0 => ("param", "Parameters to be normalized"),
    1 => ("indices", "Sparse indices, only needed for sparse param"),
    2 => ("grad", "Gradient computed, only needed for sparse param")
}

outputs!{EnsureClipped, 
    0 => ("output_param", "param ensured to be clipped within range")
}

allow_inplace!{EnsureClipped, vec![(0, 0)]}

should_not_do_gradient!{EnsureClipped}
