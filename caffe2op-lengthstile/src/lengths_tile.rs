crate::ix!();

/**
  | Given DATA tensor of rank r >= 1, and LENGTHS
  | tensor of rank 1, duplicate each entry
  | of the outer-most dimension of DATA
  | according to LENGTHS, and concatenate
  | them in an output tensor of rank r.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LengthsTileOp<Context> {
    storage:            OperatorStorage,
    context:            Context,
    lengths_host:       Tensor, // default = CPU
    row_mapping_host:   Tensor,
    row_mapping_device: Tensor,
}

register_cpu_operator!{LengthsTile, LengthsTileOp<CPUContext>}

num_inputs!{LengthsTile, 2}

num_outputs!{LengthsTile, 1}

inputs!{LengthsTile, 
    0 => ("DATA",    "Tensor of rank r >= 1. First dimension must be equal to the size of lengths"),
    1 => ("LENGTHS", "Tensor of int32 lengths of rank 1")
}

outputs!{LengthsTile, 
    0 => ("OUTPUT", "Tensor of rank r")
}

input_tags!{
    LengthsTileOp {
        Data,
        Lengths
    }
}

impl<Context> LengthsTileOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
}
