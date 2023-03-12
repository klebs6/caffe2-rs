crate::ix!();

/**
  | BinaryElementwiseOp is a wrapper around
  | BinaryElementwiseWithArgsOp, with the
  | difference that it takes a functor with
  | default constructor, e.g. that does not need
  | to take into consideration any arguments
  | during operator creation.
  */
pub type BinaryElementwiseOp<InputTypes, Context, Functor, TypeMap = SameTypeAsInput> 
= BinaryElementwiseWithArgsOp<
InputTypes,
Context,
BinaryFunctorWithDefaultCtor<Functor>,
TypeMap>;

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct BinaryElementwiseWithArgsOp<InputTypes, Context, Functor, OutputTypeMap> {
    storage:          OperatorStorage,
    context:          Context,
    legacy_broadcast: bool,
    axis:             i32,
    axis_str:         String,
    order:            String,
    functor:          Functor,
    phantomIT:        PhantomData<InputTypes>,
    phantomOTM:       PhantomData<OutputTypeMap>,
}

impl<InputTypes,Context,Functor,OutputTypeMap> BinaryElementwiseWithArgsOp<InputTypes, Context, Functor, OutputTypeMap> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(bool, "broadcast", legacy_broadcast_, false),
            OP_SINGLE_ARG(int, "axis", axis_, -1),
            OP_SINGLE_ARG(string, "axis_str", axis_str_, string("")),
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
            return DispatchHelper<InputTypes>::call(this, Input(0));
        */
    }

    #[inline] pub fn do_run_with_type<T>() -> bool {
        todo!();
        /*
            const auto& A = Input(0);
            const auto& B = Input(1);

            const T* A_data = A.template data<T>();
            const T* B_data = B.template data<T>();
            std::vector<int> A_dims;
            std::vector<int> B_dims;
            std::vector<int64_t> C_dims;

            if (legacy_broadcast_) {
              CAFFE_ENFORCE(
                  !IsInputOutputAlias(1, 0),
                  "In-place is allowed only with the first tensor when "
                  "legacy-broadcasting");
              C_dims = A.sizes().vec();
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
              // TODO: change the types to vector<int64_t>
              auto C_dims_int =
                  elementwise_ops_utils::ComputeBinaryBroadcastForwardDims(
                      A_dims, B_dims);
              std::copy(
                  C_dims_int.cbegin(), C_dims_int.cend(), std::back_inserter(C_dims));
              if (IsInputOutputAlias(0, 0)) {
                CAFFE_ENFORCE_EQ(C_dims_int, A_dims);
              } else if (IsInputOutputAlias(1, 0)) {
                CAFFE_ENFORCE_EQ(C_dims_int, B_dims);
              }
            }

            auto* C = Output(
                0, C_dims, at::dtype<typename OutputTypeMap::template type<T>>());
            auto* C_data =
                C->template mutable_data<typename OutputTypeMap::template type<T>>();
            return functor_.Forward(A_dims, B_dims, A_data, B_data, C_data, &context_);
        */
    }
}

