crate::ix!();

///---------------------------------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct CosineSimilarityGradientOp<T, Context> {

    storage: OperatorStorage,
    context: Context,
    aux:     Tensor,
    phantom: PhantomData<T>,
}

num_inputs!{CosineSimilarityGradient, 3}

num_outputs!{CosineSimilarityGradient, 2}

pub struct GetCosineSimilarityGradient;

impl GetGradientDefs for GetCosineSimilarityGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "CosineSimilarityGradient",
            "",
            vector<string>{I(0), I(1), GO(0)},
            vector<string>{GI(0), GI(1)});
        */
    }
}

register_cpu_operator!{
    CosineSimilarityGradient,
    CosineSimilarityGradientOp<f32, CPUContext>
}

register_gradient!{CosineSimilarity, GetCosineSimilarityGradient}

impl<T, Context> CosineSimilarityGradientOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
}

input_tags!{
    CosineSimilarityGradientOp {
        XIn,
        YIn,
        DerCosIn
    }
}

output_tags!{
    CosineSimilarityGradientOp {
        DerXOut,
        DerYOut
    }
}
