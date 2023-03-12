crate::ix!();

pub struct MulFunctor<Context> {
    //.FillUsing(MathDocGenerator("multiplication", kMulExample))
    phantom: PhantomData<Context>,
}

num_inputs!{Mul, 2}

num_outputs!{Mul, 1}

cost_inference_function!{Mul, /* (PointwiseCostInference<1>) */ }

tensor_inference_function!{Mul, /* (ElementwiseOpShapeInference) */}

allow_inplace!{Mul, vec![(0, 0), (1, 0)]}

inherit_onnx_schema!{Mul}

///-----------------------

num_inputs!{MulGradient, 3}

num_outputs!{MulGradient, 2}

tensor_inference_function!{MulGradient, /* (ElementwiseGradientOpShapeInference) */}

allow_inplace!{MulGradient, vec![(0, 0), (0, 1)]}

///-----------------------

impl<Context> MulFunctor<Context> {

    #[inline] pub fn forward<TIn, TOut>(
        &self, 
        a_dims:    &Vec<i32>,
        b_dims:    &Vec<i32>,
        a:         *const TIn,
        b:         *const TIn,
        c:         *mut TOut,
        context:   *mut Context) -> bool 
    {
        todo!();
        /*
            math::Mul(
                A_dims.size(),
                A_dims.data(),
                B_dims.size(),
                B_dims.data(),
                A,
                B,
                C,
                context);
            return true;
        */
    }
}

register_cpu_operator!{
    Mul,
    BinaryElementwiseOp<NumericTypes, CPUContext, MulFunctor<CPUContext>>
}

register_cpu_operator!{
    MulGradient,
    BinaryElementwiseGradientOp<
        NumericTypes,
        CPUContext,
        MulFunctor<CPUContext>>
}
