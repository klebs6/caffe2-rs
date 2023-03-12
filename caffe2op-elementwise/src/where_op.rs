crate::ix!();

/**
  | Operator Where takes three input data
  | (Tensor, Tensor, Tensor) and produces one output
  | data (Tensor) where z = c ? x : y is applied
  | elementwise.
  |
  */
#[USE_OPERATOR_FUNCTIONS(Context)]
#[USE_DISPATCH_HELPER]
pub struct WhereOp<Context> {

    storage: OperatorStorage,
    context: Context,

    enable_broadcast: bool,

    /*
      | Input: C, X, Y,
      | 
      | output: Z
      |
      */
}

num_inputs!{Where, 3}

num_outputs!{Where, 1}

inputs!{Where, 
    0 => ("C", "input tensor containing booleans"),
    1 => ("X", "input tensor"),
    2 => ("Y", "input tensor")
}

outputs!{Where, 
    0 => ("Z", "output tensor")
}

identical_type_and_shape_of_input!{Where, 1}

allow_inplace!{Where, vec![(1, 2)]}

impl<Context> WhereOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(bool, "broadcast_on_rows", enable_broadcast_, 0)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<
            TensorTypes<float, double, int, long, std::string, bool>>::
            call(this, Input(1));
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            auto& select = Input(0);
            auto& left = Input(1);
            auto& right = Input(2);

            if (enable_broadcast_) {
              CAFFE_ENFORCE_EQ(select.dim(), 1);
              CAFFE_ENFORCE_EQ(select.size(0), right.size(0));
              CAFFE_ENFORCE_EQ(left.sizes(), right.sizes());
            } else {
              CAFFE_ENFORCE_EQ(select.sizes(), left.sizes());
              CAFFE_ENFORCE_EQ(select.sizes(), right.sizes());
            }
            auto* output = Output(0, left.sizes(), at::dtype<T>());

            const bool* select_data = select.template data<bool>();
            const T* left_data = left.template data<T>();
            const T* right_data = right.template data<T>();
            T* output_data = output->template mutable_data<T>();

            if (enable_broadcast_) {
              size_t block_size = left.size_from_dim(1);
              for (int i = 0; i < select.numel(); i++) {
                size_t offset = i * block_size;
                if (select_data[i]) {
                  context_.CopyItemsSameDevice(
                      output->dtype(),
                      block_size,
                      left_data + offset,
                      output_data + offset);
                } else {
                  context_.CopyItemsSameDevice(
                      output->dtype(),
                      block_size,
                      right_data + offset,
                      output_data + offset);
                }
              }
            } else {
              for (int i = 0; i < select.numel(); ++i) {
                output_data[i] = select_data[i] ? left_data[i] : right_data[i];
              }
            }
            return true;
        */
    }
}
