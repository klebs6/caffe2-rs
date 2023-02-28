// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/q8vadd.cc]

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod arch_x86 {

    use super::*;

    #[test] fn q8vadd_sse2_n_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          VAddMicrokernelTester().n(8).test(pytorch_q8vadd_ukernel__sse2);

        */
    }

    #[test] fn q8vadd_sse2_n_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 8; n < 128; n += 24) {
            VAddMicrokernelTester().n(n).test(pytorch_q8vadd_ukernel__sse2);
          }

        */
    }

    #[test] fn q8vadd_sse2_n_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 9; n < 16; n++) {
            VAddMicrokernelTester().n(n).test(pytorch_q8vadd_ukernel__sse2);
          }

        */
    }

    #[test] fn q8vadd_sse2_n_lt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 8; n++) {
            VAddMicrokernelTester().n(n).test(pytorch_q8vadd_ukernel__sse2);
          }

        */
    }

    #[test] fn q8vadd_sse2_inplace_a() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 128; n += 11) {
            VAddMicrokernelTester().iterations(1).n(n).inplaceA(true).test(
                pytorch_q8vadd_ukernel__sse2);
          }

        */
    }

    #[test] fn q8vadd_sse2_inplace_b() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 128; n += 11) {
            VAddMicrokernelTester().iterations(1).n(n).inplaceB(true).test(
                pytorch_q8vadd_ukernel__sse2);
          }

        */
    }

    #[test] fn q8vadd_sse2_inplace_a_and_b() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 128; n += 11) {
            VAddMicrokernelTester()
                .iterations(1)
                .n(n)
                .inplaceA(true)
                .inplaceB(true)
                .test(pytorch_q8vadd_ukernel__sse2);
          }

        */
    }

    #[test] fn q8vadd_sse2_a_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 128; n += 11) {
            for (float aScale = 1.0e-2; aScale < 1.0e+2; aScale *= 1.7f) {
              VAddMicrokernelTester().iterations(1).n(n).aScale(aScale).test(
                  pytorch_q8vadd_ukernel__sse2);
            }
          }

        */
    }

    #[test] fn q8vadd_sse2_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 128; n += 11) {
            for (float bScale = 1.0e-2; bScale < 1.0e+2; bScale *= 1.7f) {
              VAddMicrokernelTester().iterations(1).n(n).bScale(bScale).test(
                  pytorch_q8vadd_ukernel__sse2);
            }
          }

        */
    }

    #[test] fn q8vadd_sse2_y_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 128; n += 11) {
            for (float yScale = 1.0e-2; yScale < 1.0e+2; yScale *= 1.7f) {
              VAddMicrokernelTester().iterations(1).n(n).yScale(yScale).test(
                  pytorch_q8vadd_ukernel__sse2);
            }
          }

        */
    }

    #[test] fn q8vadd_sse2_a_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 128; n += 11) {
            for (i32 aZeroPoint = 0; aZeroPoint <= 255; aZeroPoint += 51) {
              VAddMicrokernelTester()
                  .iterations(1)
                  .n(n)
                  .aZeroPoint(u8(aZeroPoint))
                  .test(pytorch_q8vadd_ukernel__sse2);
            }
          }

        */
    }

    #[test] fn q8vadd_sse2_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 128; n += 11) {
            for (i32 bZeroPoint = 0; bZeroPoint <= 255; bZeroPoint += 51) {
              VAddMicrokernelTester()
                  .iterations(1)
                  .n(n)
                  .bZeroPoint(u8(bZeroPoint))
                  .test(pytorch_q8vadd_ukernel__sse2);
            }
          }

        */
    }

    #[test] fn q8vadd_sse2_y_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 128; n += 11) {
            for (i32 yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
              VAddMicrokernelTester()
                  .iterations(1)
                  .n(n)
                  .yZeroPoint(u8(yZeroPoint))
                  .test(pytorch_q8vadd_ukernel__sse2);
            }
          }

        */
    }

    #[test] fn q8vadd_sse2_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 128; n += 11) {
            VAddMicrokernelTester().iterations(1).n(n).qmin(128).test(
                pytorch_q8vadd_ukernel__sse2);
          }

        */
    }

    #[test] fn q8vadd_sse2_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 128; n += 11) {
            VAddMicrokernelTester().iterations(1).n(n).qmax(128).test(
                pytorch_q8vadd_ukernel__sse2);
          }

        */
    }
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
pub mod arch_arm {

    use super::*;

    #[test] fn q8vadd_neon_n_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          VAddMicrokernelTester().n(8).test(pytorch_q8vadd_ukernel__neon);

        */
    }

    #[test] fn q8vadd_neon_n_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 8; n < 128; n += 24) {
            VAddMicrokernelTester().n(n).test(pytorch_q8vadd_ukernel__neon);
          }

        */
    }

    #[test] fn q8vadd_neon_n_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 9; n < 16; n++) {
            VAddMicrokernelTester().n(n).test(pytorch_q8vadd_ukernel__neon);
          }

        */
    }

    #[test] fn q8vadd_neon_n_lt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 8; n++) {
            VAddMicrokernelTester().n(n).test(pytorch_q8vadd_ukernel__neon);
          }

        */
    }

    #[test] fn q8vadd_neon_inplace_a() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 128; n += 11) {
            VAddMicrokernelTester().iterations(1).n(n).inplaceA(true).test(
                pytorch_q8vadd_ukernel__neon);
          }

        */
    }

    #[test] fn q8vadd_neon_inplace_b() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 128; n += 11) {
            VAddMicrokernelTester().iterations(1).n(n).inplaceB(true).test(
                pytorch_q8vadd_ukernel__neon);
          }

        */
    }

    #[test] fn q8vadd_neon_inplace_a_and_b() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 128; n += 11) {
            VAddMicrokernelTester()
                .iterations(1)
                .n(n)
                .inplaceA(true)
                .inplaceB(true)
                .test(pytorch_q8vadd_ukernel__neon);
          }

        */
    }

    #[test] fn q8vadd_neon_a_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 128; n += 11) {
            for (float aScale = 1.0e-2; aScale < 1.0e+2; aScale *= 1.7f) {
              VAddMicrokernelTester().iterations(1).n(n).aScale(aScale).test(
                  pytorch_q8vadd_ukernel__neon);
            }
          }

        */
    }

    #[test] fn q8vadd_neon_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 128; n += 11) {
            for (float bScale = 1.0e-2; bScale < 1.0e+2; bScale *= 1.7f) {
              VAddMicrokernelTester().iterations(1).n(n).bScale(bScale).test(
                  pytorch_q8vadd_ukernel__neon);
            }
          }

        */
    }

    #[test] fn q8vadd_neon_y_scale() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 128; n += 11) {
            for (float yScale = 1.0e-2; yScale < 1.0e+2; yScale *= 1.7f) {
              VAddMicrokernelTester().iterations(1).n(n).yScale(yScale).test(
                  pytorch_q8vadd_ukernel__neon);
            }
          }

        */
    }

    #[test] fn q8vadd_neon_a_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 128; n += 11) {
            for (i32 aZeroPoint = 0; aZeroPoint <= 255; aZeroPoint += 51) {
              VAddMicrokernelTester()
                  .iterations(1)
                  .n(n)
                  .aZeroPoint(u8(aZeroPoint))
                  .test(pytorch_q8vadd_ukernel__neon);
            }
          }

        */
    }

    #[test] fn q8vadd_neon_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 128; n += 11) {
            for (i32 bZeroPoint = 0; bZeroPoint <= 255; bZeroPoint += 51) {
              VAddMicrokernelTester()
                  .iterations(1)
                  .n(n)
                  .bZeroPoint(u8(bZeroPoint))
                  .test(pytorch_q8vadd_ukernel__neon);
            }
          }

        */
    }

    #[test] fn q8vadd_neon_y_zero_point() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 128; n += 11) {
            for (i32 yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
              VAddMicrokernelTester()
                  .iterations(1)
                  .n(n)
                  .yZeroPoint(u8(yZeroPoint))
                  .test(pytorch_q8vadd_ukernel__neon);
            }
          }

        */
    }

    #[test] fn q8vadd_neon_qmin() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 128; n += 11) {
            VAddMicrokernelTester().iterations(1).n(n).qmin(128).test(
                pytorch_q8vadd_ukernel__neon);
          }

        */
    }

    #[test] fn q8vadd_neon_qmax() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 128; n += 11) {
            VAddMicrokernelTester().iterations(1).n(n).qmax(128).test(
                pytorch_q8vadd_ukernel__neon);
          }

        */
    }
}

