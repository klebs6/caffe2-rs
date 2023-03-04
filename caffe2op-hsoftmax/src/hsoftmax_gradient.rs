crate::ix!();

///------------------------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct HSoftmaxGradientOp<T, Context> {

    base: HSoftmaxOpBase<T, Context>,

    phantom: PhantomData<T>,
}

num_inputs!{HSoftmaxGradient, 6}

num_outputs!{HSoftmaxGradient, 4}

register_cpu_operator!{HSoftmax,                HSoftmaxOp<f32, CPUContext>}
register_cpu_operator!{HSoftmaxGradient,        HSoftmaxGradientOp<f32, CPUContext>}
register_cpu_operator!{HSoftmaxSearch,          HSoftmaxSearchOp<f32, CPUContext>}
register_cpu_operator!{HuffmanTreeHierarchy,    HuffmanTreeHierarchyOp<int64_t, CPUContext>}

pub struct GetHSoftmaxGradient;

impl GetGradientDefs for GetHSoftmaxGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
                "HSoftmaxGradient", "",
                //X, W, b, label, intermediate output, dY
                vector<string>{I(0), I(1), I(2), I(3), O(1), GO(0)},
                //dX, dW, db, dintermediate_output
                vector<string>{GI(0), GI(1), GI(2), GO(1)});
        */
    }
}

register_gradient!{HSoftmax, GetHSoftmaxGradient}
