crate::ix!();

#[derive(PartialEq,Eq)]
pub enum RecurrentParamOpMode { 
    SET_PARAM, 
    GET_PARAM 
}

#[USE_RECURRENT_BASE_FUNCTIONS]
pub struct RecurrentOp<T> {
    base:    RecurrentBaseOp<T>,
    phantom: PhantomData<T>,
}

input_tags!{
    RecurrentOp {
        Input,
        HiddenInput,
        CellInput,
        Weight
    }
}

output_tags!{
    RecurrentOp {
        Output,
        HiddenOutput,
        CellOutput,
        RnnScratch,
        DropoutStates
    }
}

impl<T> RecurrentOp<T> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : RecurrentBaseOp<T>(operator_def, ws)
        */
    }
}
