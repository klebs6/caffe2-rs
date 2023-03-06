crate::ix!();

/**
  | Given a tensor of int32 `lengths` tensor
  | representing segment lengths and a
  | `mask` (boolean) tensor, return the
  | segment lengths of the corresponding
  | segmented tensor after *BooleanMask**
  | is applied.
  | 
  | If `lengths` tensor is $[a_1, a_2, ...,
  | a_n]$, then length of `mask` tensor
  | must be $a_1 + a_2 + ... + a_n$.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/boolean_mask_ops.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct BooleanMaskLengthsOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{BooleanMaskLengths, 2}

num_outputs!{BooleanMaskLengths, 1}

inputs!{
    BooleanMaskLengths, 
    0 => ("lengths", "(*Tensor`<int>`*): input tensor containing segment lengths"),
    1 => ("mask", "(*Tensor`<bool>`*): A 1D bool tensor of values to keep.")
}

outputs!{
    BooleanMaskLengths, 
    0 => ("masked_lengths", "(*Tensor`<int>`*): 1D tensor of same type as inputs that contains the sequence")
}

impl<Context> BooleanMaskLengthsOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(0));
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            auto& lengths = Input(0);
            auto& mask = Input(1);

            CAFFE_ENFORCE(lengths.dim() == 1);
            CAFFE_ENFORCE(mask.dim() == 1);
            const auto* lengthsPtr = lengths.template data<T>();
            const auto* maskPtr = mask.template data<bool>();
            auto totalLength =
                std::accumulate(lengthsPtr, lengthsPtr + lengths.numel(), 0);
            CAFFE_ENFORCE(mask.numel() == totalLength);
            auto* lengthsOut = Output(0, lengths.sizes(), at::dtype<T>());
            auto* lengthsOutPtr = lengthsOut->template mutable_data<T>();
            int p = 0;
            for (int i = 0; i < lengths.numel(); ++i) {
              T lengthOut = 0;
              for (int j = 0; j < lengthsPtr[i]; ++j) {
                if (maskPtr[p++]) {
                  ++lengthOut;
                }
              }
              lengthsOutPtr[i] = lengthOut;
            }
            return true;
        */
    }
}
