crate::ix!();

pub struct SinhFunctor<Context> {
    phantom: PhantomData<Context>,
}

impl<Context> SinhFunctor<Context> {
    
    #[inline] pub fn invoke<T>(&self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
            math::Sinh(N, X, Y, context);
        return true;
        */
    }
}

/**
  | Calculates the hyperbolic sine of the
  | given input tensor, element-wise.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sinh_op.cc
  |
  */
register_cpu_operator!{Sinh,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        SinhFunctor<CPUContext>>}

num_inputs!{Sinh, 1}

num_outputs!{Sinh, 1}

inputs!{Sinh, 
    0 => ("input", "Input tensor")
}

outputs!{Sinh, 
    0 => ("output", "The hyperbolic sine values of the input tensor, computed element-wise")
}

identical_type_and_shape!{Sinh}

inherit_onnx_schema!{Sinh}
