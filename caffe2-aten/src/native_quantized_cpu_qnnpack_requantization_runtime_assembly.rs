// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/runtime-assembly.h]

lazy_static!{
    /*
    #ifdef __aarch64__

    .macro SUB_ZERO_POINT vout, vin1, vin2
    #if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
        USUBL \vout, \vin1, \vin2
    #else
        UXTL \vout, \vin1
    #endif
    .endm

    #else /* aarch32 */

    .macro SUB_ZERO_POINT qout, din1, din2
    #if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
        VSUBL.U8 \qout, \din1, \din2
    #else
        VMOVL.U8 \qout, \din1
    #endif
    .endm

    #endif
    */
}

