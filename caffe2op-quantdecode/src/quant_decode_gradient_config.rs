crate::ix!();

num_outputs!{QuantDecodeGradient, 1}

num_inputs!{QuantDecodeGradient, 
    |input: i32| {
        input >= 3 && input % 2 == 1
    }
}

num_inputs_outputs!{QuantDecode, 
    |input: i32, output: i32| {
        input > 1 && output + 1 == input 
    }
}

inputs!{QuantDecode, 
    0 => ("codebook", "Codebook in 1d tensor (float)"),
    1 => ("codes_0", "Encoded codes 0 (uint8/uint16/int32)"),
    2 => ("codes_1", "Encoded codes 1 if existed (uint8/uint16/int32)"),
    3 => ("codes_n", "Encoded codes n if existed (uint8/uint16/int32)")
}

outputs!{QuantDecode, 
    0 => ("decoded_0", "Decoded tensor for codes_0 (float)"),
    1 => ("decoded_1", "Decoded tensor for codes_1 (float)"),
    2 => ("decoded_n", "Decoded tensor for codes_n (float)")
}

register_cpu_operator!{QuantDecode, QuantDecodeOp<QuantDecodeRunTy::RUN_ALWAYS>}

register_cpu_gradient_operator!{QuantDecodeGradient, QuantDecodeGradientOp}

#[cfg(caffe2_use_mpscnn)]
register_cpu_operator!{ MPSCNNQuantDecode, QuantDecodeOp<QuantDecodeRunTy::RUN_ONCE>}
