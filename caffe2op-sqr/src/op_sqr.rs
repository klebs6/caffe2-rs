crate::ix!();

/**
  | Performs element-wise squaring ($x^2$)
  | of input tensor.
  | 
  | Github Link:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sqr_op.cc
  |
  */
pub struct SqrFunctor<Context> {
    
    phantom: PhantomData<Context>,
}

impl<Context> SqrFunctor<Context> {
    
    #[inline] pub fn invoke<T>(&self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool {
    
        todo!();
        /*
            math::Sqr(N, X, Y, context);
        return true;
        */
    }
}

register_cpu_operator!{
    Sqr,
    UnaryElementwiseOp< 
        TensorTypes<f32>, 
        CPUContext, 
        SqrFunctor<CPUContext>>
}

num_inputs!{Sqr, 1}

num_outputs!{Sqr, 1}

inputs!{Sqr, 
    0 => ("X", "*(type: Tensor`<float>`)* Input data tensor.")
}

outputs!{Sqr, 
    0 => ("Y", "*(type: Tensor`<float>`)* Output tensor.")
}

identical_type_and_shape!{Sqr}

allow_inplace!{Sqr, vec![(0, 0)]}


register_cuda_operator!{Sqr,
    UnaryElementwiseOp<
        TensorTypes<f32>,
        CUDAContext,
        SqrFunctor<CUDAContext>>
}
