crate::ix!();

pub struct BinaryElementwiseDivGradientWithArgsGradientOp 
{
    base: BinaryElementwiseWithArgsGradientOp<
        NumericTypes,
        CPUContext,
        BinaryFunctorWithDefaultCtor::<DivFunctor::<CPUContext>>, 
        SameTypeAsInput, 
        SameTypeAsInput>,

    storage: OperatorStorage,
    context: CPUContext,

    legacy_broadcast: bool,
    axis:             i32,
    axis_str:         String,
    order:            String,
    functor: BinaryFunctorWithDefaultCtor<DivFunctor<CPUContext>>,
}

impl BinaryElementwiseDivGradientWithArgsGradientOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...),
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
            return DispatchHelper<NumericTypes>::call(this, Input(1));
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            const T* dC_data = nullptr;
            const T* A_data = nullptr;
            const T* B_data = nullptr;
            const T* C_data = nullptr;
            std::vector<int> A_dims;
            std::vector<int> B_dims;
            at::IntArrayRef dA_sizes;
            at::IntArrayRef dB_sizes;
            if (InputSize() == 3) {
              const auto& B = Input(0);
              const auto& C = Input(1);
              const auto& dC = Input(2);
              if (legacy_broadcast_) {
                if (B.numel() == 1) {
                  A_dims = {static_cast<int>(C.numel())};
                  B_dims = {1};
                } else {
                  size_t pre, n, post;
                  std::tie(pre, n, post) =
                      elementwise_ops_utils::ComputeLegacyBroadcastSizes(C, B, axis_);
                  A_dims = {static_cast<int>(pre),
                            static_cast<int>(n),
                            static_cast<int>(post)};
                  B_dims = {static_cast<int>(n), 1};
                }
              } else {
                std::copy(
                    C.sizes().cbegin(), C.sizes().cend(), std::back_inserter(A_dims));
                std::copy(
                    B.sizes().cbegin(), B.sizes().cend(), std::back_inserter(B_dims));
              }
              B_data = B.template data<T>();
              C_data = C.template data<T>();
              dC_data = dC.template data<T>();
              dA_sizes = C.sizes();
              dB_sizes = B.sizes();
            } else {
              const auto& dC = Input(0);
              const auto& A = Input(1);
              const auto& B = Input(2);
              const auto& C = Input(3);
              if (legacy_broadcast_) {
                if (B.numel() == 1) {
                  A_dims = {static_cast<int>(A.numel())};
                  B_dims = {1};
                } else {
                  size_t pre, n, post;
                  std::tie(pre, n, post) =
                      elementwise_ops_utils::ComputeLegacyBroadcastSizes(A, B, axis_);
                  A_dims = {static_cast<int>(pre),
                            static_cast<int>(n),
                            static_cast<int>(post)};
                  B_dims = {static_cast<int>(n), 1};
                }
              } else {
                std::copy(
                    A.sizes().cbegin(), A.sizes().cend(), std::back_inserter(A_dims));
                std::copy(
                    B.sizes().cbegin(), B.sizes().cend(), std::back_inserter(B_dims));
              }
              dC_data = dC.template data<T>();
              A_data = A.template data<T>();
              B_data = B.template data<T>();
              C_data = C.template data<T>();
              dA_sizes = A.sizes();
              dB_sizes = B.sizes();
            }
            auto* dA = Output(0, dA_sizes, at::dtype<T>());
            auto* dB = Output(1, dB_sizes, at::dtype<T>());
            auto* dA_data = dA->template mutable_data<T>();
            auto* dB_data = dB->template mutable_data<T>();
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

register_cpu_operator!{
    DivGradient,
    BinaryElementwiseGradientOp<
        NumericTypes,
        CPUContext,
        DivFunctor<CPUContext>>
}

pub struct GetDivGradient;

impl GetGradientDefs for GetDivGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "DivGradient",
            "",
            std::vector<std::string>{GO(0), I(0), I(1), O(0)},
            std::vector<std::string>{GI(0), GI(1)});
        */
    }
}

register_gradient!{Div, GetDivGradient}
