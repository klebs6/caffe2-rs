crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/assembly.h]

lazy_static!{
    /*
    #ifdef __ELF__
        .macro BEGIN_FUNCTION name
            .text
            .align 2
            .global \name
            .type \name, %function
            \name:
        .endm

        .macro END_FUNCTION name
            .size \name, .-\name
        .endm
    #elif defined(__MACH__)
        .macro BEGIN_FUNCTION name
            .text
            .align 2
            .global _\name
            .private_extern _\name
            _\name:
        .endm

        .macro END_FUNCTION name
        .endm
    #endif
    */
}

