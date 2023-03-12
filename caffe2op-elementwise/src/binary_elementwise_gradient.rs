crate::ix!();

/**
  | BinaryElementwiseGradientOp is a wrapper
  | around BinaryElementwiseGradientWithArgsOp,
  | with the difference that it takes a functor
  | with default constructor, e.g. that does not
  | need to take into consideration any arguments
  | during operator creation.
  */
pub type BinaryElementwiseGradientOp<InputTypes, Context, Functor, OutputTypeMap, GradientTypeMap> 
= BinaryElementwiseWithArgsGradientOp<
InputTypes,
Context,
BinaryFunctorWithDefaultCtor<Functor>,
OutputTypeMap,
GradientTypeMap>;

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct BinaryElementwiseWithArgsGradientOp<InputTypes, Context, Functor, OutputTypeMap, GradientTypeMap> {
    storage:                OperatorStorage,
    context:                Context,
    legacy_broadcast:       bool,
    axis:                   i32,
    axis_str:               String,
    order:                  String,
    functor:                Functor,
    phantomIT:              PhantomData<InputTypes>,
    phantomOTM:             PhantomData<OutputTypeMap>,
    phantomGradientTypeMap: PhantomData<GradientTypeMap>,
}

impl<InputTypes, Context, Functor, OutputTypeMap, GradientTypeMap> 
BinaryElementwiseWithArgsGradientOp<InputTypes, Context, Functor, OutputTypeMap, GradientTypeMap> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(bool, "broadcast", legacy_broadcast_, false),
            OP_SINGLE_ARG(int, "axis", axis_, -1),
            OP_SINGLE_ARG(string, "axis_str", axis_str_, ""),
            OP_SINGLE_ARG(string, "order", order_, "NCHW"),
            functor_(*this) 

        if (legacy_broadcast_) {
          if (axis_ != -1) {
            // Get axis from an explicit axis argument.
            CAFFE_ENFORCE_EQ(
                axis_str_.size(),
                0U,
                "Args axis and axis_str cannot be used simultaneously.");
          } else if (axis_str_.size()) {
            // Get the axis index semantically.
            CAFFE_ENFORCE_EQ(
                axis_str_.size(), 1U, "Unsupported axis string", axis_str_);
            const size_t semantic_axis_ = order_.find(axis_str_);
            CAFFE_ENFORCE_NE(
                semantic_axis_,
                string::npos,
                "Unrecognizable axis string ",
                axis_str_,
                " from order string ",
                order_);
            axis_ = semantic_axis_;
          } else {
            CAFFE_ENFORCE(
                axis_ == -1 && axis_str_.empty(),
                "Do not specify axis or axis_str if broadcast is not enabled.");
          }
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<InputTypes>::call(this, Input(1));
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            const auto& dC = Input(0);
            const auto& A = Input(1);
            const auto& B = Input(2);

            vector<int> A_dims;
            vector<int> B_dims;
            if (legacy_broadcast_) {
              if (B.numel() == 1) {
                A_dims = {static_cast<int>(A.numel())};
                B_dims = {1};
              } else {
                size_t pre, n, post;
                std::tie(pre, n, post) =
                    elementwise_ops_utils::ComputeLegacyBroadcastSizes(A, B, axis_);
                A_dims = {
                    static_cast<int>(pre), static_cast<int>(n), static_cast<int>(post)};
                B_dims = {static_cast<int>(n), 1};
              }
            } else {
              std::copy(
                  A.sizes().cbegin(), A.sizes().cend(), std::back_inserter(A_dims));
              std::copy(
                  B.sizes().cbegin(), B.sizes().cend(), std::back_inserter(B_dims));
            }
            const typename OutputTypeMap::template type<T>* C_data = nullptr;
            if (InputSize() == 4) {
              const auto& C = Input(3);
              C_data = C.template data<typename OutputTypeMap::template type<T>>();
            }
            const auto* dC_data =
                dC.template data<typename GradientTypeMap::template type<T>>();
            const T* A_data = A.template data<T>();
            const T* B_data = B.template data<T>();
            auto* dA = Output(
                0, A.sizes(), at::dtype<typename GradientTypeMap::template type<T>>());
            auto* dB = Output(
                1, B.sizes(), at::dtype<typename GradientTypeMap::template type<T>>());
            auto* dA_data =
                dA->template mutable_data<typename GradientTypeMap::template type<T>>();
            auto* dB_data =
                dB->template mutable_data<typename GradientTypeMap::template type<T>>();
            return functor_.Backward(
                A_dims,
                B_dims,
                dC_data,
                A_data,
                B_data,
                C_data,
                dA_data,
                dB_data,
                &context_);
        */
    }
}
