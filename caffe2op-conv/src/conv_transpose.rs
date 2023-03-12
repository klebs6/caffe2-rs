crate::ix!();

/**
 | The ConvTranspose op takes an input data tensor
 | $X$, an input weight tensor $filter$, and
 | optionally an input bias tensor $bias$. 
 |
 | It then computes the transposed convolution,
 | sometimes referred to as deconvolution, and
 | produces a single output tensor $Y$. The
 | hyperparameters of the op such as kernel size,
 | stride, and padding are specified as args. 
 |
 | At each stride, the filter is deconvolved with
 | a subset of $X$ and the $bias$ is added. This is
 | done throughout the input data until the output
 | computation is complete.
 |
 | The output shapes are computed as follows. The
 | number of channels in the output feature map is
 | the number of kernels specified in the filter
 | blob. 
 |
 | The spatial height and width are computed as
 |
 | $$H_{out} = (H_{in}-1)*strides[0] - 2*pads[0] + kernels[0]$$
 |
 |
 | $$W_{out} = (W_{in}-1)*strides[1] - 2*pads[1] + kernels[1]$$
 |
 | Note on the implementation layout:
 | conv_transpose_op_impl.h is the templated
 | implementation of the conv_transpose_op.h file,
 | which is why they are separate files. Also, in the
 | implementation this operator inherits from the
 | *ConvTransposeUnpoolOpBase* operator.
 |
 | Github Links:
 | - https://github.com/pytorch/pytorch/tree/master/caffe2/operators/conv_transpose_op.h
 | - https://github.com/pytorch/pytorch/tree/master/caffe2/operators/conv_transpose_op.cc
 | - https://github.com/pytorch/pytorch/tree/master/caffe2/operators/conv_transpose_unpool_op_base.h
 |
 */
#[USE_CONV_TRANSPOSE_UNPOOL_BASE_FUNCTIONS("Context")]
pub struct ConvTransposeOp<T, Context> {
    base:            ConvTransposeUnpoolBase<Context>,
    col_buffer:      Tensor,
    bias_multiplier: Tensor,

    // Input: X, W, b
    //
    // Output: Y
    //
    phantom:         PhantomData<T>,
}

num_inputs!{ConvTranspose, (2,3)}

num_outputs!{ConvTranspose, 1}

inputs!{ConvTranspose, 
    0 => ("X",               "Input data blob, of shape $(N, C_{in}, H_{in}, W_{in})$, to be operated on."),
    1 => ("filter",          "The filter blob, of shape $(M, C_{out}, K_H, K_W)$, containing the filters to be used in the transposed convolution."),
    2 => ("bias",            "The bias blob, of length $C_{out}$, containing the biases for the operation, one bias per output channel. If not passed, biases assumed to be zeros.")
}

outputs!{ConvTranspose, 
    0 => ("Y",               "Output data blob, of shape $(N, C_{out}, H_{out}, W_{out})$, that contains the result of the operation.")
}

args!{ConvTranspose, 
    0 => ("legacy_pad",      "*(type: int; optional)* Should the legacy padding be VALID or SAME. When used, pads should not be used."),
    1 => ("kernels",         "*(type: [int]; default: [])* Desired kernel size. If left at default the kernel size will be inferred from the input $filter$ blob."),
    2 => ("strides",         "*(type: [int]; default: [])* Controls the stride of the kernel as it traverses the input blob."),
    3 => ("pads",            "*(type: [int]; default: [])* Controls the amount of padding applied to the input feature map before computation."),
    4 => ("adjs",            "*(type: [int]; default: [])*"),
    5 => ("order",           "*(type: string; default: NCHW)* Specifies the order of the input data blob, where $N$ is batch size, $C$ is number of channels, $H$ is spatial height, and $W$ is spatial width. The only other valid option is NHWC."),
    6 => ("shared_buffer",   "*(type: int; default: 0)*"),
    7 => ("no_bias",         "*(type: bool; default: False)* ")
}

inherit_onnx_schema!{ConvTranspose}

register_cpu_operator!{
    ConvTranspose, 
    ConvTransposeOp<f32, CPUContext>
}

input_tags!{
    ConvTransposeOp {
        Input,
        Filter,
        Bias
    }
}

impl<T,Context> ConvTransposeOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : ConvTransposeUnpoolBase<Context>(std::forward<Args>(args)...)
        */
    }
}
