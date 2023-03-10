// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/q8dwconv/up8x9-aarch32-neon.S]

lazy_static!{
    /*
    #include <qnnpack/assembly.h>
    #include <requantization/runtime-assembly.h>

    .syntax unified

    # void pytorch_q8dwconv_ukernel_up8x9__aarch32_neon(
    #     usize channels,
    #     usize output_width,
    #     const u8** input,
    #     const void* weights,
    #     u8* output,
    #     usize input_stride,
    #     usize output_increment,
    #     const union pytorch_qnnp_conv_quantization_params quantization_params[restrict static 1])
    BEGIN_FUNCTION pytorch_q8dwconv_ukernel_up8x9__aarch32_neon
        .arm
    #ifndef __APPLE__
        .arch armv7-a
        .fpu neon
    #endif

        # Load params
        # - r12 = quantization_params
        LDR r12, [sp, 12]

        PUSH {r4, r5, r6, r7, r8, r9, r10, r11, lr}
        VPUSH {d8-d15}

        STR r0, [sp, #-8]
        STR r3, [sp, #-4]

        # Load the address zero_point array.
        # For depth wise kernels the array is of single element.
        LDR r5, [r12], 4

        # Load o:
        # - lr = o = output
        LDR lr, [sp, 100]

        # Load kernel zero point:
        # - d31 = vkernel_zero_point
        VLD1.8 {d31[]}, [r5]

        # Load input zero point:
        # - d30 = vinput_zero_point
        VLD1.8 {d30[]}, [r12]
        # Load the address requantization_scale array.
        # For depth wise kernels the array is of single element.
        # pre-index r12 = r12 + 4
        LDR r5, [r12, 4]!

        # add 8 bytes to get to vfmax
        ADD r12, r12, 8
        # Load requantization_scale:
        # - q14 = d28:d29 = requantization_scale
        VLD1.32 {d28[], d29[]}, [r5]

        # Load vfmax:
        # - q13 = d26:d27 = vfmax
        VLD1.32 {d26[], d27[]}, [r12]!

        # Load vfmin:
        # - q12 = d24:d25 = vfmin
        VLD1.32 {d24[], d25[]}, [r12]!

        # Load vfmagic:
        # - q10 = d20:d21 = vfmagic
        VLD1.32 {d20[], d21[]}, [r12]!

        # Load vimagic:
        # - q11 = d22:d23 = vimagic
        # Since q11/d22 gets used in the remainder channels section
        # This load will have to occur in that section again.
        # But since r12 is overwritten below, we will have to push it
        # on the stack and pop it back.
        VLD1.32 {d22[], d23[]}, [r12]

        VSTR d22, [sp, #-16]
        VSTR d23, [sp, #-24]

        .p2align 3
    0:
        # Load input stride
        # - r3 = input_stride
        LDR r3, [sp, 104]

        # Load c:
        # - r0 = c = channels
        LDR r0, [sp, #-8]

        # Load i0, i1, i2, i3, i4, i5, i6, i7, i8
        # - r4 = i0
        # - r5 = i1
        # - r6 = i2
        # - r7 = i3
        # - r8 = i4
        # - r9 = i5
        # - r10 = i6
        # - r11 = i7
        # - r12 = i8
        LDM r2, {r4, r5, r6, r7, r8, r9, r10, r11, r12}

        # Pre-decrement c
        SUBS r0, r0, 8

        # Increment input by input stride
        # - input = r2 := input + input_stride
        ADD r2, r2, r3

        # Load w:
        # - r3 = w = weights
        LDR r3, [sp, #-4]

        BLO 2f

        .p2align 4
    1:
        VLDM r3!, {d0-d3}

        VLD1.8 {d4}, [r4]!
        VLD1.8 {d6}, [r3]!

        VLD1.8 {d8}, [r5]!
        VLD1.8 {d10}, [r3]!

        SUB_ZERO_POINT q2, d4, d30
        VSUBL.U8 q3, d6, d31

        VLD1.8 {d12}, [r6]!
        VLD1.8 {d14}, [r3]!

        SUB_ZERO_POINT q4, d8, d30
        VSUBL.U8 q5, d10, d31

        VMLAL.S16 q0, d4, d6
        VMLAL.S16 q1, d5, d7

        VLD1.8 {d4}, [r7]!
        VLD1.8 {d6}, [r3]!

        SUB_ZERO_POINT q6, d12, d30
        VSUBL.U8 q7, d14, d31

        VMLAL.S16 q0, d8, d10
        VMLAL.S16 q1, d9, d11

        VLD1.8 {d8}, [r8]!
        VLD1.8 {d10}, [r3]!

        SUB_ZERO_POINT q2, d4, d30
        VSUBL.U8 q3, d6, d31

        VMLAL.S16 q0, d12, d14
        VMLAL.S16 q1, d13, d15

        VLD1.8 {d12}, [r9]!
        VLD1.8 {d14}, [r3]!

        SUB_ZERO_POINT q4, d8, d30
        VSUBL.U8 q5, d10, d31

        VMLAL.S16 q0, d4, d6
        VMLAL.S16 q1, d5, d7

        VLD1.8 {d4}, [r10]!
        VLD1.8 {d6}, [r3]!

        SUB_ZERO_POINT q6, d12, d30
        VSUBL.U8 q7, d14, d31

        VMLAL.S16 q0, d8, d10
        VMLAL.S16 q1, d9, d11

        VLD1.8 {d8}, [r11]!
        VLD1.8 {d10}, [r3]!

        SUB_ZERO_POINT q2, d4, d30
        VSUBL.U8 q3, d6, d31

        VMLAL.S16 q0, d12, d14
        VMLAL.S16 q1, d13, d15

        VLD1.8 {d12}, [r12]!
        VLD1.8 {d14}, [r3]!

        SUB_ZERO_POINT q4, d8, d30
        VSUBL.U8 q5, d10, d31

        VMLAL.S16 q0, d4, d6
        VMLAL.S16 q1, d5, d7

        SUB_ZERO_POINT q6, d12, d30
        VSUBL.U8 q7, d14, d31

        VMLAL.S16 q0, d8, d10
        VMLAL.S16 q1, d9, d11

        VMLAL.S16 q0, d12, d14
        VMLAL.S16 q1, d13, d15

        VCVT.F32.S32 q0, q0
        VCVT.F32.S32 q1, q1

        VMUL.F32 q0, q0, q14
        VMUL.F32 q1, q1, q14

        VMIN.F32 q0, q0, q13
        VMIN.F32 q1, q1, q13

        VMAX.F32 q0, q0, q12
        VMAX.F32 q1, q1, q12

        VADD.F32 q0, q0, q10
        VADD.F32 q1, q1, q10

        VSUB.S32 q0, q0, q11
        VSUB.S32 q1, q1, q11

        VQMOVN.S32 d0, q0
        VQMOVN.S32 d1, q1
        VQMOVUN.S16 d0, q0

        VST1.8 {d0}, [lr]!
        SUBS r0, r0, 8
        BHS 1b

    2:
        CMP r0, -8
        BEQ 5f

        ADD r4, r4, r0
        ADD r5, r5, r0
        ADD r6, r6, r0
        ADD r7, r7, r0
        ADD r8, r8, r0
        ADD r9, r9, r0
        ADD r10, r10, r0
        ADD r11, r11, r0
        ADD r12, r12, r0

        LSL r0, r0, 3
        VDUP.32 d22, r0

        VLDM r3!, {d0-d3}

        VLD1.8 {d4}, [r4]!
        VLD1.8 {d6}, [r3]!
        VLD1.8 {d8}, [r5]!
        VLD1.8 {d10}, [r3]!

        VSHL.U64 d4, d4, d22

        VLD1.8 {d12}, [r6]!
        VLD1.8 {d14}, [r3]!

        SUB_ZERO_POINT q2, d4, d30
        VSUBL.U8 q3, d6, d31

        VSHL.U64 d8, d8, d22

        VLD1.8 {d16}, [r7]!
        VLD1.8 {d18}, [r3]!

        VSHL.U64 d12, d12, d22

        SUB_ZERO_POINT q4, d8, d30
        VSUBL.U8 q5, d10, d31

        VMLAL.S16 q0, d4, d6
        VMLAL.S16 q1, d5, d7

        VLD1.8 {d4}, [r8]!
        VLD1.8 {d6}, [r3]!

        VSHL.U64 d16, d16, d22

        SUB_ZERO_POINT q6, d12, d30
        VSUBL.U8 q7, d14, d31

        VMLAL.S16 q0, d8, d10
        VMLAL.S16 q1, d9, d11

        VLD1.8 {d8}, [r9]!
        VLD1.8 {d10}, [r3]!

        VSHL.U64 d4, d4, d22

        SUB_ZERO_POINT q8, d16, d30
        VSUBL.U8 q9, d18, d31

        VMLAL.S16 q0, d12, d14
        VMLAL.S16 q1, d13, d15

        VLD1.8 {d12}, [r10]!
        VLD1.8 {d14}, [r3]!

        VSHL.U64 d8, d8, d22

        SUB_ZERO_POINT q2, d4, d30
        VSUBL.U8 q3, d6, d31

        VMLAL.S16 q0, d16, d18
        VMLAL.S16 q1, d17, d19

        VLD1.8 {d16}, [r11]!
        VLD1.8 {d18}, [r3]!

        VSHL.U64 d12, d12, d22

        SUB_ZERO_POINT q4, d8, d30
        VSUBL.U8 q5, d10, d31

        VMLAL.S16 q0, d4, d6
        VMLAL.S16 q1, d5, d7

        VLD1.8 {d4}, [r12]!
        VLD1.8 {d6}, [r3]!

        VSHL.U64 d16, d16, d22

        SUB_ZERO_POINT q6, d12, d30
        VSUBL.U8 q7, d14, d31

        VMLAL.S16 q0, d8, d10
        VMLAL.S16 q1, d9, d11

        VSHL.U64 d4, d4, d22

        SUB_ZERO_POINT q8, d16, d30
        VSUBL.U8 q9, d18, d31

        VMLAL.S16 q0, d12, d14
        VMLAL.S16 q1, d13, d15

        SUB_ZERO_POINT q2, d4, d30
        VSUBL.U8 q3, d6, d31

        VMLAL.S16 q0, d16, d18
        VMLAL.S16 q1, d17, d19

        VMLAL.S16 q0, d4, d6
        VMLAL.S16 q1, d5, d7

        VLDR.64 d22, [sp, #-16]
        VLDR.64 d23, [sp, #-24]

        VCVT.F32.S32 q0, q0
        VCVT.F32.S32 q1, q1

        VMUL.F32 q0, q0, q14
        VMUL.F32 q1, q1, q14

        VMIN.F32 q0, q0, q13
        VMIN.F32 q1, q1, q13

        VMAX.F32 q0, q0, q12
        VMAX.F32 q1, q1, q12

        VADD.F32 q0, q0, q10
        VADD.F32 q1, q1, q10

        VSUB.S32 q0, q0, q11
        VSUB.S32 q1, q1, q11

        VQMOVN.S32 d0, q0
        VQMOVN.S32 d1, q1
        VQMOVUN.S16 d0, q0


        TST r0, 32
        BEQ 3f
        VST1.32 {d0[0]}, [lr]!
        VEXT.8 d0, d0, 4

    3:
        TST r0, 16
        BEQ 4f
        VST1.16 {d0[0]}, [lr]!
        VEXT.8 d0, d0, 2

    4:
        TST r0, 8
        BEQ 5f
        VST1.8 {d0[0]}, [lr]!

    5:
        # Load output increment
        # - r3 = output_increment
        LDR r3, [sp, 108]

        # Decrement output width
        SUBS r1, r1, 1

        # Increment output by output_increment
        ADD lr, lr, r3

        # If output width is non-zero, process another pixel
        BNE 0b

        VPOP {d8-d15}
        POP {r4, r5, r6, r7, r8, r9, r10, r11, pc}
    END_FUNCTION pytorch_q8dwconv_ukernel_up8x9__aarch32_neon

    #ifdef __ELF__
    .section ".note.GNU-stack","",%progbits
    #endif
    */
}

