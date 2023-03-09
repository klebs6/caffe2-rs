crate::ix!();

/**
  | bool FirstDim = true>
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct AbstractReduceFrontOrBackGradientOp<T,Context,ReducerGradient,const FirstDim: bool> {
    storage:                OperatorStorage,
    context:                Context,
    num_reduce_dims:        i32,
    phantom:                PhantomData<T>,
    phantomReducerGradient: PhantomData<ReducerGradient>,
}

pub enum _InputTags {
    REDUCTION_GRAD,// = <ReducerGradient as HasOriginalInputs>::original_inputs_size(),
    SOURCE_SHAPE,
}

impl<T,Context,ReducerGradient: HasOriginalInputs,const FirstDim: bool> AbstractReduceFrontOrBackGradientOp<T,Context,ReducerGradient,FirstDim> {

    const kNumInputs: isize = todo!(); // <ReducerGradient as HasOriginalInputs>::original_inputs_size() + 2;
    
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
            // If more complicated fixed size logic becomes necessary, it can be moved
        // to the reducer class
        int64_t grad_block_size = Input(REDUCTION_GRAD).numel();
        return DispatchHelper<typename ReducerGradient::FixedDispatch>::call(
            this, grad_block_size);
        */
    }
    
    #[inline] pub fn do_run_with_value<const FixedSize: i32>(&mut self) -> bool {
    
        todo!();
        /*
            auto& reduction_grad = Input(REDUCTION_GRAD);
        auto& source_shape = this->template Input<Tensor>(SOURCE_SHAPE, CPU);

        typename ReducerGradient::Meta ctx(reduction_grad, 0, FirstDim);
        for (int i = 0; i < ReducerGradient::originalInputs().size(); ++i) {
          auto& aux_in = Input(i);
          ctx.observeOriginalInput(
              ReducerGradient::originalInputs()[i],
              aux_in,
              nullptr, /*no grad*/
              num_reduce_dims_);
        }

        const T* r_grad = reduction_grad.template data<T>();

        CAFFE_ENFORCE_LE(num_reduce_dims_, source_shape.numel());

        vector<int64_t> shape(
            source_shape.template data<int64_t>(),
            source_shape.template data<int64_t>() + source_shape.numel());

        auto* data_grads = Output(0, shape, at::dtype<T>());

        int64_t block_size = FirstDim
            ? data_grads->size_from_dim(num_reduce_dims_)
            : data_grads->size_from_dim(data_grads->dim() - num_reduce_dims_);
        int64_t block_num = block_size > 0 ? data_grads->numel() / block_size : 0;

        T* out = data_grads->template mutable_data<T>();

        ReducerGradient r(ctx, r_grad, &context_);
        for (int64_t i = 0; i < block_num; ++i) {
          r.template fillGrad<FixedSize>(
              ctx,
              out + block_size * i,
              i,
              &context_,
              FirstDim ? block_num : block_size);
        }
        return true;
        */
    }
}
