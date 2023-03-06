crate::ix!();

/**
  | Given a 1D `data` tensor and a boolean
  | `mask` tensor of the same shape, returns
  | a `masked_data` tensor containing
  | only the elements corresponding to
  | positions where the `mask` is True,
  | and a `masked_indices` tensor containing
  | the indices of the True elements.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/boolean_mask_ops.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct BooleanMaskOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{BooleanMask, 2}

num_outputs!{BooleanMask, (1,2)}

inputs!{
    BooleanMask, 
    0 => ("data", "(*Tensor*): 1D input tensor"),
    1 => ("mask", "(*Tensor`<bool>`*): tensor of bools which determines the input elements that will be left in the `masked_data` output tensor; same shape as `data`")
}

outputs!{
    BooleanMask, 
    0 => ("masked_data", "(*Tensor*): 1D tensor of same type as `data` input that contains the masked input tensor"),
    1 => ("masked_indices", "(*Tensor`<int>`*): 1D tensor of indices of the True elements in the `mask` tensor")
}

impl<Context> BooleanMaskOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
}
