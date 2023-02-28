crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/u8rmax.cc]

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod arch_arm {

    use super::*;

    #[test] fn U8RMAX__NEON_n_lt_16() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 16; n++) {
            RMaxMicrokernelTester().n(n).test(pytorch_u8rmax_ukernel__neon);
          }

        */
    }

    #[test] fn U8RMAX__NEON_n_eq_16() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          RMaxMicrokernelTester().n(16).test(pytorch_u8rmax_ukernel__neon);

        */
    }

    #[test] fn U8RMAX__NEON_n_div_16() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 16; n < 128; n += 16) {
            RMaxMicrokernelTester().n(n).test(pytorch_u8rmax_ukernel__neon);
          }

        */
    }

    #[test] fn U8RMAX__NEON_n_gt_16() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 16; n < 32; n++) {
            RMaxMicrokernelTester().n(n).test(pytorch_u8rmax_ukernel__neon);
          }

        */
    }

}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod arch_x86 {

    use super::*;

    #[test] fn U8RMAX__SSE2_n_lt_16() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 16; n++) {
            RMaxMicrokernelTester().n(n).test(pytorch_u8rmax_ukernel__sse2);
          }

        */
    }

    #[test] fn U8RMAX__SSE2_n_eq_16() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          RMaxMicrokernelTester().n(16).test(pytorch_u8rmax_ukernel__sse2);

        */
    }

    #[test] fn U8RMAX__SSE2_n_div_16() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 16; n < 128; n += 16) {
            RMaxMicrokernelTester().n(n).test(pytorch_u8rmax_ukernel__sse2);
          }

        */
    }

    #[test] fn U8RMAX__SSE2_n_gt_16() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 17; n < 32; n++) {
            RMaxMicrokernelTester().n(n).test(pytorch_u8rmax_ukernel__sse2);
          }

        */
    }
}
