crate::ix!();

/**
  | The Conv2D operator computes a 2D convolution
  | operation over an input blob $(X)$,
  | with a filter blob $(filter)$ and a bias
  | blob $(bias)$, and outputs a single
  | output blob $(Y)$.
  | 
  | Although there are several options
  | for order, the convention is that the
  | input $(X)$ is a blob of shape $(N,C_{in},H_{in},W_{in})$
  | and the output $(Y)$ is a blob of shape
  | $(N,C_{out},H_{out},W_{out})$.
  | Here, $N$ is the batch size, $C$ is the
  | number of channels, $H$ is the spatial
  | height, and $W$ is the spatial width.
  | For example, if your input data was a
  | batch of five, 100x120pixel RGB images,
  | $X$ would have shape $(5,3,120,100)$.
  | 
  | The $filter$ input blob may contain
  | multiple filters and has shape $(M,
  | C_{in}, K_H, K_W)$.
  | 
  | Here, $M$ is the number of individual
  | filters contained in the blob, $C_{in}$
  | is the number of channels of each filter
  | (by convention in 2D convolution it
  | is the same as the number of channels
  | in the input), $K_H$ is the spatial height
  | of the kernel, and $K_W$ is the spatial
  | width of the kernel.
  | 
  | The $bias$ blob is a vector of length
  | $M$, where there is one bias for each
  | filter in the $filter$ blob.
  | 
  | Given the shape of the input blob and
  | the filter blob, we can calculate the
  | shape of the output blob as follows.
  | The number of items in the batch $N$ will
  | stay the same. The number of channels
  | in the output will equal the number of
  | kernels in the filter blob, so $C_{out}
  | = M.$ With stride and pad defined below,
  | the spatial height and width of the output
  | ($H_{out}$ and $W_{out}$) are calculated
  | as
  | 
  | $$H_{out} = \left \lfloor{\frac{H_{in}
  | - K_H + 2*pad}{stride}+1}\right \rfloor$$
  | 
  | $$W_{out} = \left \lfloor{\frac{W_{in}
  | - K_W + 2*pad}{stride}+1}\right \rfloor$$
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_op.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_op.cc
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h
  |
  */
#[USE_CONV_POOL_BASE_FUNCTIONS("Context")]
pub struct ConvOp<T, Context> {
    base:                    ConvPoolOpBase<Context>,
    col_buffer:              Tensor, // {Context::GetDeviceType()};
    bias_multiplier:         Tensor, // {Context::GetDeviceType()};
    img_shape_device:        Tensor, // {Context::GetDeviceType()};
    col_buffer_shape_device: Tensor, // {Context::GetDeviceType()};
    phantom:                 PhantomData<T>,
}

impl<T,Context> ConvOp<T,Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : ConvPoolOpBase<Context>(operator_def, ws) 

        // Since this is the default convolution implementation, we will
        // use CAFFE_ENFORCE instead of OPERATOR_NEEDS_FEATURE.
        CAFFE_ENFORCE(
            (group_ == 1 || order_ == StorageOrder::NCHW ||
             std::is_same<Context, CPUContext>::value),
            "Group convolution only supports NCHW order or CPUContext right now.");

        // Create shared buffer mutex in the constructor
        // to avoid race-condition in DAGNet.
        if (FLAGS_caffe2_force_shared_col_buffer || shared_buffer_) {
          createSharedBuffer<Context>(ws_);
        }
        */
    }
}

/// Input: X, W, b
/// Output: Y
input_tags!{
    ConvOp {
        Input,
        Filter,
        Bias
    }
}
