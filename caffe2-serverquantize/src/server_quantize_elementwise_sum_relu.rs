crate::ix!();

use crate::{
    SumOp,
    OperatorDef,
    Workspace
};

pub struct SumReluOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS
    base: SumOp<Context>,
}

impl<Context> SumReluOp<Context> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : SumOp<Context>(operator_def, ws)
        */
    }
    
    #[inline] pub fn do_run_with_type<T, M>(&mut self) -> bool {
    
        todo!();
        /*
            if (!SumOp<Context>::template DoRunWithType<T>()) {
          return false;
        }

        auto* output = Output(0);
        T* output_data = output->template mutable_data<T>();
        for (int i = 0; i < output->size(); ++i) {
          output_data[i] = std::max(static_cast<T>(0), output_data[i]);
        }
        return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (Input(0).template IsType<float>()) {
          return DoRunWithType<float, float>();
        } else if (Input(0).template IsType<double>()) {
          return DoRunWithType<double, double>();
        } else if (Input(0).template IsType<int>()) {
          return DoRunWithType<int, int>();
        } else {
          CAFFE_THROW(
              "Sum operator only supports 32-bit float, 64-bit double and ints, but",
              " input was of type ",
              Input(0).dtype().name());
        }
        */
    }
}

register_cpu_operator!{SumRelu, SumReluOp<CPUContext>}
