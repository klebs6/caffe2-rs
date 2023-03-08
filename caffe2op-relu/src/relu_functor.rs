crate::ix!();

/**
 | Applies rectified linear unit operation to the
 | input data element-wise. The Relu operation takes
 | one input $X$, produces one output $Y$, and is
 | defined as:
 |
 | $$Y = max(0,X)$$
 |
 | Github Links:
 | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/relu_op.h
 | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/relu_op.cc
 |
 */
pub struct ReluFunctor<Context> {
    
    /**
      | Input: X
      | 
      | output: Y
      |
      */
    phantom: PhantomData<Context>,
}

num_inputs!{Relu, 1}

num_outputs!{Relu, 1}

inputs!{Relu, 
    0 => ("X", "1D input tensor")
}

outputs!{Relu, 
    0 => ("Y", "1D output tensor with same shape as input")
}

identical_type_and_shape!{Relu}

cost_inference_function!{Relu, CostInferenceForRelu }

allow_inplace!{Relu, vec![(0, 0)]}

inherit_onnx_schema!{Relu}

impl ReluFunctor<CPUContext> {

    #[inline] pub fn invoke<T>(&self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut CPUContext) -> bool {
    
        todo!();
        /*
            EigenVectorMap<T>(Y, N) = ConstEigenVectorMap<float>(X, N).cwiseMax(T(0));
      return true;
        */
    }
    
    #[cfg(caffe2_use_accelerate)]
    #[inline] pub fn invoke_f32(&self, 
        n:       i32,
        x:       *const f32,
        y:       *mut f32,
        context: *mut CPUContext) -> bool {
        
        todo!();
        /*
          const float zero = 0.0f;
          vDSP_vthres(X, 1, &zero, Y, 1, N);
          return true;
        */
    }
}
