crate::ix!();

#[inline] pub fn int_8dequantize(
    input:    *const u8,
    out:      *mut f32,
    n:        i64,
    x_scale:  f32,
    x_offset: i32)  {
    
    todo!();
    /*
        for (auto i = 0; i < N; ++i) {
        out[i] = (static_cast<int32_t>(in[i]) - X_offset) * X_scale;
      }
    */
}

pub struct Int8DequantizeOp {
    storage: OperatorStorage,
    context: CPUContext,
}

num_inputs!{Int8Dequantize, 1}

num_outputs!{Int8Dequantize, 1}

inputs!{Int8Dequantize, 
    0 => ("qX", "Int8 Tensor qX.")
}

outputs!{Int8Dequantize, 
    0 => ("Y", "FP32 Tensor that represents mapped real value of qX.")
}

identical_type_and_shape!{Int8Dequantize}

impl Int8DequantizeOp {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Inputs()[0]->template Get<Int8TensorCPU>();

        auto* Y = Output(0, X.t.sizes(), at::dtype<float>());
        int32_t X_offset = X.zero_point;
        auto X_scale = X.scale;
        Int8Dequantize(
            X.t.data<uint8_t>(),
            Y->mutable_data<float>(),
            X.t.numel(),
            X_scale,
            X_offset);
        return true;
        */
    }
}

register_cpu_operator!{Int8Dequantize, int8::Int8DequantizeOp}
