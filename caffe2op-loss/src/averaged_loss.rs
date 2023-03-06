crate::ix!();

/**
  | The *AveragedLoss* op takes a single
  | 1-D input tensor *input* and returns
  | a single output float value *output*.
  | The output represents the average of
  | the values in *input*. This op is commonly
  | used for averaging losses, hence the
  | name, however it does not exclusively
  | operate on losses.
  | 
  | Github Links:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/loss_op.h
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/loss_op.cc
  | 
  | AveragedLoss takes in the input and
  | produces the output loss value as the
  | average of the input.
  |
  */
pub struct AveragedLoss<T, Context> {
    base: SumElementsOp<T, Context>,

    phantom: PhantomData<T>,
}

num_inputs!{AveragedLoss, 1}

num_outputs!{AveragedLoss, 1}

inputs!{AveragedLoss, 
    0 => ("input", "The input data as Tensor")
}

outputs!{AveragedLoss, 
    0 => ("output", "The output tensor of size 1 containing the averaged value.")
}

scalar_type!{AveragedLoss, TensorProto::FLOAT}


impl<T, Context> AveragedLoss<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : SumElementsOp<T, Context>(std::forward<Args>(args)..., true)
        */
    }
}

