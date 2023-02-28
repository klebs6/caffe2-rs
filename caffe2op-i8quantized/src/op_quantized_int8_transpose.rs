crate::ix!();

use crate::{
    OperatorDef,
    Workspace,
    TransposeOp,
    CPUContext,
};

/**
  | Transpose the input tensor by permuting
  | the axes of the input according to the
  | `axes` argument.
  | 
  | Similar to numpy's [transpose](https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html)
  | function.
  | 
  | For example, when axes=(1, 0, 2), given
  | an input tensor of shape
  | 
  | (1, 2, 3), the output shape will be (2,
  | 1, 3).
  |
  */
pub struct Int8TransposeOp {
    base: TransposeOp<CPUContext>,
}

num_inputs!{Int8Transpose, 1}

num_outputs!{Int8Transpose, 1}

inputs!{Int8Transpose, 
    0 => ("X", "Input tensor")
}

outputs!{Int8Transpose, 
    0 => ("Y", "Transposed output")
}

args!{Int8Transpose, 
    0 => ("axes",          "*(type: Tuple(int))* Order to permute axes of input tensor. Reverses the dimensions by default."),
    1 => ("Y_scale",       "Output tensor quantization scale"),
    2 => ("Y_zero_point",  "Output tensor quantization offset")
}

register_cpu_operator!{Int8Transpose, Int8TransposeOp}

impl Int8TransposeOp {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : TransposeOp(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Inputs()[0]->Get<Int8TensorCPU>();
        auto* Y = Outputs()[0]->GetMutable<Int8TensorCPU>();
        int32_t Y_zero_point =
            this->template GetSingleArgument<int>("Y_zero_point", 0);
        auto Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
        CAFFE_ENFORCE_EQ(Y_zero_point, X.zero_point);
        CAFFE_ENFORCE_EQ(Y_scale, X.scale);
        Y->scale = Y_scale;
        Y->zero_point = Y_zero_point;
        TransposeImpl<uint8_t>(X.t, &Y->t);
        return true;
        */
    }
}

