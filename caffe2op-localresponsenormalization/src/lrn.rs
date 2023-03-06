crate::ix!();

/**
  | `LRN` applies Local Response Normalization
  | to an input blob. This operation performs
  | a kind of "lateral inhibition" by normalizing
  | over local input regions, where normalization
  | is applied across channels. This operator
  | is typically used to normalize an unbounded
  | activation (such as ReLU). The output
  | shape is the same as the input shape.
  | The `brew` module has a wrapper for this
  | operator for use in a `ModelHelper`
  | object.
  | 
  | The formula for LRN is as follows:
  | 
  | $$b_{c} = a_{c}(bias + \frac{\alpha}{n}\sum_{c'=max(0,c-n/2)}^{min(N-1,c+n/2)}
  | a_{c'}^2 )^{-\beta}$$
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/local_response_normalization_op.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/local_response_normalization_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LRNOp<T, Context> {

    base: LRNOpBase<T, Context>,

    scale:              *mut Tensor, // = nullptr;
    local_scale_tensor: Tensor,      //{Context::GetDeviceType()};

    /**
      | Input: X;
      | 
      | Output: Y, scale.
      |
      */
    phantom: PhantomData<T>,
}

num_inputs!{LRN, 1}

num_outputs!{LRN, (1,2)}

inputs!{LRN, 
    0 => ("X", "*(type: Tensor`<float>`)* Input data tensor (ReLU output).")
}

outputs!{LRN, 
    0 => ("Y", "*(type: Tensor`<float>`)* Output tensor."),
    1 => ("Y_scale", "*(type: Tensor`<float>`)* Output scale.")
}

args!{LRN, 
    0 => ("size", "*(type: int; default: 0)* Amount of neighboring channels to sum over for normalization"),
    1 => ("alpha", "*(type: float; default: 0)* Multiplicative (scaling) factor."),
    2 => ("beta", "*(type: float; default: 0)* Exponent."),
    3 => ("bias", "*(type: float; default: 1.0)* Additive factor."),
    4 => ("order", "*(type: float; default: 'NCHW')* Order of blob dimensions.")
}

inherit_onnx_schema!{LRN}

output_tags!{
    LRNOp {
        Output,
        Scale
    }
}

impl<T, Context> LRNOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : LRNOpBase<T, Context>(std::forward<Args>(args)...)
        */
    }
}

