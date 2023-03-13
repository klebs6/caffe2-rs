crate::ix!();

/**
  | @brief
  | 
  | Simple non-segmented reduction over
  | the first few dimensions of the tensor
  | 
  | Inputs:
  | 
  | 0: DATA - input embedding to do lookups
  | in
  | 
  | 1..P: AUX_ARG_<I> - optional additional
  | arguments to be passed to the reducer
  | 
  | Args: num_reduce_dim (default 1) -
  | the number of dims in front of the tensor
  | to reduce
  | 
  | Output:
  | 
  | Tensor without the first `num_dim`
  | dimensions of DATA class InputAccessor
  | = BaseInputAccessor<T>>
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct AbstractReduceFrontOrBackOp<T,Context,Reducer,const FirstDim: bool,InputAccessor> {
    storage:         OperatorStorage,
    context:         Context,
    num_reduce_dims: i32,
    input_accessor:  InputAccessor,
    phantom:         PhantomData<T>,
    phantomReducer:  PhantomData<Reducer>,
}

impl<T,Context,R: Reducer,const FirstDim: bool,InputAccessor> 
AbstractReduceFrontOrBackOp<T,Context,R,FirstDim,InputAccessor> {

    const kNumInputs: isize = <R as Reducer>::InputCount;

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "num_reduce_dim", num_reduce_dims_, 1)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& data = Input(0);
        // If more complicated fixed size logic becomes necessary, it can be moved
        // to the reducer class
        int64_t in_block_size = FirstDim
            ? data.size_from_dim(num_reduce_dims_)
            : data.size_to_dim(data.dim() - num_reduce_dims_);
        return DispatchHelper<typename Reducer::FixedDispatch>::call(
            this, in_block_size);
        */
    }
    
    #[inline] pub fn do_run_with_value<const FixedSize: i32>(&mut self) -> bool {
    
        todo!();
        /*
            auto& data = Input(0);

        CAFFE_ENFORCE_LE(num_reduce_dims_, data.dim());

        typename Reducer::Meta ctx(FirstDim);
        ctx.observeInput(0, data, num_reduce_dims_);
        for (int i = 1; i < <R as Reducer>::InputCount; ++i) {
          auto& aux_in = Input(i);
          ctx.observeInput(i, aux_in, num_reduce_dims_);
        }

        OPERATOR_NEEDS_FEATURE(
            inputAccessor_.observeInput(data),
            "Unsupported input type: ",
            data.dtype().name(),
            ".");

        vector<int64_t> shape;
        ctx.appendOutputShape(&shape);
        auto* output = Output(0, shape, at::dtype<T>());

        T* out = output->template mutable_data<T>();

        const int block_size = FirstDim
            ? data.size_from_dim(num_reduce_dims_)
            : data.size_from_dim(data.dim() - num_reduce_dims_);

        const int num_blocks = block_size > 0 ? data.numel() / block_size : 0;

        Reducer r(ctx, out, &context_);
        for (int64_t i = 0; i < num_blocks; ++i) {
          r.template process<FixedSize>(
              ctx, inputAccessor_.getBlockPtr(block_size, i), i, &context_);
        }
        r.template finish<FixedSize>(ctx, &context_);
        return true;
        */
    }
}
