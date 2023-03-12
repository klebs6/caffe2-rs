crate::ix!();

/**
 | This op implements the exponential linear unit
 | (ELU) activation function as described in [Fast
 | and Accurate 
 |
 | Deep Network Learning by Exponential Linear Units
 | (ELUs)]
 |
 | (https://arxiv.org/abs/1511.07289). 
 |
 | The op takes an input tensor $X$ of arbitrary
 | shape, computes the elementwise elu operation, and
 | returns a vector $Y$ of the same shape as output. 
 |
 | The alpha parameter may be passed as an argument,
 | but defaults to 1. 
 |
 | The elu operation is defined as
 |
 | $$y=f(x) =\begin{cases}\alpha(e^x-1) & x < 0 \\
 | x & otherwise\end{cases}$$
 |
 | Github Links:
 | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elu_op.h
 | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elu_op.cc
 */
pub struct EluFunctor<Context> {
    alpha: f32,

    /**
      | Input: X,
      | 
      | output: Y
      |
      */
    phantom: PhantomData<Context>,
}

num_inputs!{Elu, 1}

num_outputs!{Elu, 1}

inputs!{Elu, 
    0 => ("X", "1D input tensor of data to be operated on.")
}

outputs!{Elu, 
    0 => ("Y", "1D input tensor, calculated as described above.")
}

args!{Elu, 
    0 => ("alpha", "*(type: float; default: 1.0)* Defines alpha parameter used in calculation.")
}

allow_inplace!{Elu, vec![(0, 0)]}

identical_type_and_shape!{Elu}

inherit_onnx_schema!{Elu}

register_cpu_operator!{
    Elu,
    UnaryElementwiseWithArgsOp<
        TensorTypes<f32>,
        CPUContext,
        EluFunctor<CPUContext>>
}

impl<Context> EluFunctor<Context> {
    
    pub fn new(op: &mut OperatorStorage) -> Self {
        todo!();
        /*
            : alpha(op.GetSingleArgument<float>("alpha", 1.0f))
        */
    }
}
