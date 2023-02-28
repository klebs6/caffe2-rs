crate::ix!();

use crate::{
    SpatialBNOp,
    OperatorDef,
    Workspace,
    CPUContext
};

pub struct SpatialBNReluOp {
    base:    SpatialBNOp<CPUContext>,
    context: CPUContext,
}

impl SpatialBNReluOp {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : SpatialBNOp<CPUContext>(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (!SpatialBNOp<CPUContext>::RunOnDevice()) {
          return false;
        }

        auto* output = Output(0);
        float* output_data = output->template mutable_data<float>();
        for (int i = 0; i < output->size(); ++i) {
          output_data[i] = std::max(0.0f, output_data[i]);
        }
        return true;
        */
    }
}

num_inputs!{SpatialBNRelu, (5,7)}

num_outputs!{SpatialBNRelu, (1,5)}

enforce_inplace!{SpatialBNRelu, vec![(3, 1), (4, 2)]}

allow_inplace!{SpatialBNRelu, vec![(0, 0), (5, 3), (6, 4)]}

register_cpu_operator!{SpatialBNRelu, SpatialBNReluOp}
