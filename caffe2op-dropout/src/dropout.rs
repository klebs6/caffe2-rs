crate::ix!();

/**
  | `Dropout` takes one input data tensor
  | (`X`) and produces two tensor outputs,
  | `Y` and `mask`.
  | 
  | If the `is_test` argument is zero (default=0),
  | the output `Y` will be the input with
  | random elements zeroed.
  | 
  | The probability that a given element
  | is zeroed is determined by the `ratio`
  | argument.
  | 
  | If the `is_test` argument is set to non-zero,
  | the output `Y` is exactly the same as
  | the input `X`.
  | 
  | -----------
  | @note
  | 
  | outputs are scaled by a factor of $\frac{1}{1-ratio}$
  | during training, so that during test
  | time, we can simply compute an identity
  | function. This scaling is important
  | because we want the output at test time
  | to equal the expected value at training
  | time.
  | 
  | Dropout has been proven to be an effective
  | regularization technique to prevent
  | overfitting during training.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dropout_op.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dropout_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct DropoutOp<T, Context> {

    storage: OperatorStorage,
    context: Context,

    ratio:   f32,
    is_test: bool,

    /**
      | Input: X;
      | 
      | Output: Y, mask.
      |
      */
    phantom: PhantomData<T>,
}

num_inputs!{Dropout, 1}

num_outputs!{Dropout, (1,2)}

inputs!{Dropout, 
    0 => ("X", "*(type: Tensor`<float>`)* Input data tensor.")
}

outputs!{Dropout, 
    0 => ("Y", "*(type: Tensor`<float>`)* Output tensor."),
    1 => ("mask", "*(type: Tensor`<bool>`)* The output mask containing boolean values for each element, signifying which elements are dropped out. If `is_test` is nonzero, this output is not filled.")
}

args!{Dropout, 
    0 => ("ratio", "*(type: float; default: 0.5)* Probability of an element to be zeroed.")
}

inherit_onnx_schema!{Dropout}

tensor_inference_function!{Dropout, /*[](const OperatorDef& def,
    const vector<TensorShape>& in) {
    CAFFE_ENFORCE_EQ(1, in.size());
    vector<TensorShape> out;
    ArgumentHelper argsHelper(def);
    out.push_back(in[0]);
    if (def.output().size() == 2) {
        out.push_back(in[0]);
        out[1].set_data_type(TensorProto_DataType_BOOL);
    }
    return out;
    }*/
}

allow_inplace!{Dropout, vec![(0, 0)]}

arg_is_test!{Dropout, 
    "*(type: int; default: 0)* If zero (train mode), perform dropout. If non-zero (test mode), Y = X."
}

impl<T, Context> DropoutOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            ratio_(this->template GetSingleArgument<float>("ratio", 0.5)),
            is_test_(this->template GetSingleArgument<int>(OpSchema::Arg_IsTest, 0)) 

        CAFFE_ENFORCE_GE(ratio_, 0);
        CAFFE_ENFORCE_LT(ratio_, 1);
        */
    }
}

