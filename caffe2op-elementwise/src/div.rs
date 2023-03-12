crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct DivFunctor<Context> {
    storage: OperatorStorage,
    context: Context,
    //.FillUsing(MathDocGenerator("division", kDivExample))
}

num_inputs!{Div, 2}

num_outputs!{Div, 1}

cost_inference_function!{Div, /* (PointwiseCostInference<1>) */ }

tensor_inference_function!{Div, /* (ElementwiseOpShapeInference) */}

allow_inplace!{Div, vec![(0, 0)]}

inherit_onnx_schema!{Div}

///--------------------------------
num_inputs!{DivGradient, (3,4)}

num_outputs!{DivGradient, 2}

tensor_inference_function!{DivGradient, /* (ElementwiseGradientOpShapeInference) */}

allow_inplace!{DivGradient, vec![(0, 0)]}

///--------------------------------

impl<Context> DivFunctor<Context> {

    #[inline] pub fn forward<TIn, TOut>(
        &mut self, 
        a_dims:    &Vec<i32>,
        b_dims:    &Vec<i32>,
        a:         *const TIn,
        b:         *const TIn,
        c:         *mut TOut,
        context:   *mut Context) -> bool 
    {
        todo!();
        /*
            math::Div(
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
    Div,
    BinaryElementwiseOp<NumericTypes, CPUContext, DivFunctor<CPUContext>>
}
