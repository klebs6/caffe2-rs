crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct BatchPermutationGradientOp<T, Context> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

/**
  | Input: indices, dY (aka "gradOutput");
  | Output: dX (aka "gradInput")
  |
  */
num_inputs!{BatchPermutationGradient, 2}

num_outputs!{BatchPermutationGradient, 1}

register_cpu_operator!{
    BatchPermutationGradient, 
    BatchPermutationGradientOp<f32, CPUContext>
}

impl<T,Context> BatchPermutationGradientOp<T,Context> {
    
    pub fn new(def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(def, ws)
        */
    }
}



pub struct GetBatchPermutationGradient;

impl GetGradientDefs for GetBatchPermutationGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "BatchPermutationGradient",
            "",
            vector<string>{I(1), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{BatchPermutation, GetBatchPermutationGradient}

pub type BatchPermutationOpFloatCPU = BatchPermutationOp<f32, CPUContext>;
