crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct L1DistanceGradientOp<T, Context> {

    storage: OperatorStorage,
    context: Context,

    /**
      | Input: X, Y, dDistance;
      | 
      | Output: dX, dY
      |
      */
    phantom: PhantomData<T>,
}

register_cpu_operator!{
    L1DistanceGradient,
    L1DistanceGradientOp<f32, CPUContext>
}

register_ideep_operator!{
    L1DistanceGradient,
    IDEEPFallbackOp::<
        L1DistanceGradientOp::<f32, CPUContext>>
}

num_inputs!{L1DistanceGradient, 3}

num_outputs!{L1DistanceGradient, 2}

pub struct GetL1DistanceGradient;

impl GetGradientDefs for GetL1DistanceGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "L1DistanceGradient",
            "",
            vector<string>{I(0), I(1), GO(0)},
            vector<string>{GI(0), GI(1)});
        */
    }
}

register_gradient!{
    L1Distance, 
    GetL1DistanceGradient
}

impl<T,Context> L1DistanceGradientOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
}
