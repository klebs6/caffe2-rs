crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/u8clamp.cc]

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod arch_arm {

    use super::*;

    #[test] fn u8clamp_neon_n_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          ClampMicrokernelTester().n(8).test(pytorch_u8clamp_ukernel__neon);

        */
    }

    #[test] fn u8clamp_neon_n_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 8; n < 512; n += 8) {
            ClampMicrokernelTester().n(n).test(pytorch_u8clamp_ukernel__neon);
          }

        */
    }

    #[test] fn u8clamp_neon_n_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 9; n < 16; n++) {
            ClampMicrokernelTester().n(n).test(pytorch_u8clamp_ukernel__neon);
          }

        */
    }

    #[test] fn u8clamp_neon_n_lt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 8; n++) {
            ClampMicrokernelTester().n(n).test(pytorch_u8clamp_ukernel__neon);
          }

        */
    }

    #[test] fn u8clamp_neon_inplace() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 128; n += 5) {
            ClampMicrokernelTester().iterations(1).n(n).inplace(true).test(
                pytorch_u8clamp_ukernel__neon);
          }

        */
    }

    #[test] fn u8clamp_neon_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 128; n += 11) {
            for (u8 qmin = 1; qmin < 255; qmin++) {
              ClampMicrokernelTester().iterations(1).n(n).qmin(qmin).test(
                  pytorch_u8clamp_ukernel__neon);
            }
          }

        */
    }

    #[test] fn u8clamp_neon_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 128; n += 11) {
            for (u8 qmax = 1; qmax < 255; qmax++) {
              ClampMicrokernelTester().iterations(1).n(n).qmax(qmax).test(
                  pytorch_u8clamp_ukernel__neon);
            }
          }

        */
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod arch_x86 {

    use super::*;

    #[test] fn u8clamp_sse2_n_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          ClampMicrokernelTester().n(8).test(pytorch_u8clamp_ukernel__sse2);

        */
    }

    #[test] fn u8clamp_sse2_n_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 8; n < 512; n += 8) {
            ClampMicrokernelTester().n(n).test(pytorch_u8clamp_ukernel__sse2);
          }

        */
    }

    #[test] fn u8clamp_sse2_n_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 9; n < 16; n++) {
            ClampMicrokernelTester().n(n).test(pytorch_u8clamp_ukernel__sse2);
          }

        */
    }

    #[test] fn u8clamp_sse2_n_lt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 8; n++) {
            ClampMicrokernelTester().n(n).test(pytorch_u8clamp_ukernel__sse2);
          }

        */
    }

    #[test] fn u8clamp_sse2_inplace() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 128; n += 5) {
            ClampMicrokernelTester().iterations(1).n(n).inplace(true).test(
                pytorch_u8clamp_ukernel__sse2);
          }

        */
    }

    #[test] fn u8clamp_sse2_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 128; n += 11) {
            for (u8 qmin = 1; qmin < 255; qmin++) {
              ClampMicrokernelTester().iterations(1).n(n).qmin(qmin).test(
                  pytorch_u8clamp_ukernel__sse2);
            }
          }

        */
    }

    #[test] fn u8clamp_sse2_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 128; n += 11) {
            for (u8 qmax = 1; qmax < 255; qmax++) {
              ClampMicrokernelTester().iterations(1).n(n).qmax(qmax).test(
                  pytorch_u8clamp_ukernel__sse2);
            }
          }

        */
    }
}
