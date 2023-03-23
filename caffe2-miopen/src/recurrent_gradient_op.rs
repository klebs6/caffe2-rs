crate::ix!();

#[USE_RECURRENT_BASE_FUNCTIONS]
pub struct RecurrentGradientOp<T> {
    base:    RecurrentBaseOp<T>,
    phantom: PhantomData<T>,
}

input_tags!{
    RecurrentGradientOp {
        Input,
        HiddenInput,
        CellInput,
        Weight,
        RnnScratch,
        Output,
        GradOutput,
        GradHiddenOutput,
        GradCellOutput
    }
}

output_tags!{
    RecurrentGradientOp {
        GradInput,
        GradHiddenInput,
        GradCellInput,
        GradWeight,
        DropoutStates,
        RnnScratchOut
    }
}

impl<T> RecurrentGradientOp<T> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : RecurrentBaseOp<T>(operator_def, ws)
        */
    }
}
