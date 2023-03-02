crate::ix!();

///---------------------------------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct DotProductGradientOp<T, Context> {
    storage: OperatorStorage,
    context: Context,

    phantom: PhantomData<T>,
}

register_cpu_operator!{
    DotProductGradient,
    DotProductGradientOp<float, CPUContext>
}

num_inputs!{DotProductGradient, 3}

num_outputs!{DotProductGradient, 2}

pub struct GetDotProductGradient;

impl GetGradientDefs for GetDotProductGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "DotProductGradient",
            "",
            vector<string>{I(0), I(1), GO(0)},
            vector<string>{GI(0), GI(1)});
        */
    }
}

register_gradient!{DotProduct, GetDotProductGradient}

impl<T,Context> DotProductGradientOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
}

input_tags!{
    DotProductGradientOp {
        XIn,
        YIn,
        DerDotIn
    }
}

output_tags!{
    DotProductGradientOp {
        DerXOut,
        DerYOut
    }
}
