crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/x8zip.cc]

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod arch_arm {

    use super::*;

    #[test] fn X8ZIP_X2__NEON_n_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          ZipMicrokernelTester().n(8).g(2).test(pytorch_qnnp_x8zip_x2__neon);

        */
    }

    #[test] fn X8ZIP_X2__NEON_n_div_16() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 8; n < 128; n += 8) {
            ZipMicrokernelTester().n(n).g(2).test(pytorch_qnnp_x8zip_x2__neon);
          }

        */
    }

    #[test] fn X8ZIP_X2__NEON_n_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 9; n < 16; n++) {
            ZipMicrokernelTester().n(n).g(2).test(pytorch_qnnp_x8zip_x2__neon);
          }

        */
    }

    #[test] fn X8ZIP_X2__NEON_n_lt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 8; n++) {
            ZipMicrokernelTester().n(n).g(2).test(pytorch_qnnp_x8zip_x2__neon);
          }

        */
    }

    #[test] fn X8ZIP_X3__NEON_n_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          ZipMicrokernelTester().n(9).g(3).test(pytorch_qnnp_x8zip_x3__neon);

        */
    }

    #[test] fn X8ZIP_X3__NEON_n_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 8; n < 128; n += 8) {
            ZipMicrokernelTester().n(n).g(3).test(pytorch_qnnp_x8zip_x3__neon);
          }

        */
    }

    #[test] fn X8ZIP_X3__NEON_n_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 9; n < 16; n++) {
            ZipMicrokernelTester().n(n).g(3).test(pytorch_qnnp_x8zip_x3__neon);
          }

        */
    }

    #[test] fn X8ZIP_X3__NEON_n_lt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 8; n++) {
            ZipMicrokernelTester().n(n).g(3).test(pytorch_qnnp_x8zip_x3__neon);
          }

        */
    }

    #[test] fn X8ZIP_X4__NEON_n_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          ZipMicrokernelTester().n(8).g(4).test(pytorch_qnnp_x8zip_x4__neon);

        */
    }

    #[test] fn X8ZIP_X4__NEON_n_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 8; n < 128; n += 8) {
            ZipMicrokernelTester().n(n).g(4).test(pytorch_qnnp_x8zip_x4__neon);
          }

        */
    }

    #[test] fn X8ZIP_X4__NEON_n_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 9; n < 16; n++) {
            ZipMicrokernelTester().n(n).g(4).test(pytorch_qnnp_x8zip_x4__neon);
          }

        */
    }

    #[test] fn X8ZIP_X4__NEON_n_lt_16() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 16; n++) {
            ZipMicrokernelTester().n(n).g(4).test(pytorch_qnnp_x8zip_x4__neon);
          }

        */
    }

    #[test] fn X8ZIP_XM__NEON_n_eq_8_m_4() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          ZipMicrokernelTester().n(8).g(4).test(pytorch_qnnp_x8zip_xm__neon);

        */
    }

    #[test] fn X8ZIP_XM__NEON_n_eq_8_m_div_4() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize g = 4; g < 32; g += 4) {
            ZipMicrokernelTester().n(8).g(g).test(pytorch_qnnp_x8zip_xm__neon);
          }

        */
    }

    #[test] fn X8ZIP_XM__NEON_n_eq_8_m_gt_4() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize g = 5; g < 8; g++) {
            ZipMicrokernelTester().n(8).g(g).test(pytorch_qnnp_x8zip_xm__neon);
          }

        */
    }

    #[test] fn X8ZIP_XM__NEON_n_div_8_m_eq_4() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 8; n < 128; n += 8) {
            ZipMicrokernelTester().n(n).g(4).test(pytorch_qnnp_x8zip_xm__neon);
          }

        */
    }

    #[test] fn X8ZIP_XM__NEON_n_div_8_m_4() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 8; n < 128; n += 8) {
            for (usize g = 4; g < 32; g += 4) {
              ZipMicrokernelTester().n(n).g(g).test(pytorch_qnnp_x8zip_xm__neon);
            }
          }

        */
    }

    #[test] fn X8ZIP_XM__NEON_n_div_8_m_gt_4() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 8; n < 128; n += 8) {
            for (usize g = 5; g < 8; g++) {
              ZipMicrokernelTester().n(n).g(g).test(pytorch_qnnp_x8zip_xm__neon);
            }
          }

        */
    }

    #[test] fn X8ZIP_XM__NEON_n_gt_8_m_eq_4() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 9; n < 16; n++) {
            ZipMicrokernelTester().n(n).g(4).test(pytorch_qnnp_x8zip_xm__neon);
          }

        */
    }

    #[test] fn X8ZIP_XM__NEON_n_gt_8_m_div_4() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 9; n < 16; n++) {
            for (usize g = 4; g < 32; g += 4) {
              ZipMicrokernelTester().n(n).g(g).test(pytorch_qnnp_x8zip_xm__neon);
            }
          }

        */
    }

    #[test] fn X8ZIP_XM__NEON_n_gt_8_m_4() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 9; n < 16; n++) {
            for (usize g = 5; g < 8; g++) {
              ZipMicrokernelTester().n(n).g(g).test(pytorch_qnnp_x8zip_xm__neon);
            }
          }

        */
    }

    #[test] fn X8ZIP_XM__NEON_n_lt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize n = 1; n < 8; n++) {
            for (usize g = 4; g < 12; g++) {
              ZipMicrokernelTester().n(n).g(g).test(pytorch_qnnp_x8zip_xm__neon);
            }
          }

        */
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod arch_x86 {

    use super::*;

    #[test] fn X8ZIP_X2__SSE2_n_eq_16() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          ZipMicrokernelTester().n(16).g(2).test(pytorch_qnnp_x8zip_x2__sse2);

        */
    }

    #[test] fn X8ZIP_X2__SSE2_n_div_16() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 16; n < 256; n += 16) {
            ZipMicrokernelTester().n(n).g(2).test(pytorch_qnnp_x8zip_x2__sse2);
          }

        */
    }

    #[test] fn X8ZIP_X2__SSE2_n_gt_16() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 17; n < 32; n++) {
            ZipMicrokernelTester().n(n).g(2).test(pytorch_qnnp_x8zip_x2__sse2);
          }

        */
    }

    #[test] fn X8ZIP_X2__SSE2_n_lt_16() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 16; n++) {
            ZipMicrokernelTester().n(n).g(2).test(pytorch_qnnp_x8zip_x2__sse2);
          }

        */
    }

    #[test] fn X8ZIP_X3__SSE2_n_eq_16() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          ZipMicrokernelTester().n(16).g(3).test(pytorch_qnnp_x8zip_x3__sse2);

        */
    }

    #[test] fn X8ZIP_X3__SSE2_n_div_16() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 16; n < 256; n += 16) {
            ZipMicrokernelTester().n(n).g(3).test(pytorch_qnnp_x8zip_x3__sse2);
          }

        */
    }

    #[test] fn X8ZIP_X3__SSE2_n_gt_16() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 17; n < 32; n++) {
            ZipMicrokernelTester().n(n).g(3).test(pytorch_qnnp_x8zip_x3__sse2);
          }

        */
    }

    #[test] fn X8ZIP_X3__SSE2_n_lt_16() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 16; n++) {
            ZipMicrokernelTester().n(n).g(3).test(pytorch_qnnp_x8zip_x3__sse2);
          }

        */
    }

    #[test] fn X8ZIP_X4__SSE2_n_eq_16() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          ZipMicrokernelTester().n(16).g(4).test(pytorch_qnnp_x8zip_x4__sse2);

        */
    }

    #[test] fn X8ZIP_X4__SSE2_n_div_16() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 16; n < 256; n += 16) {
            ZipMicrokernelTester().n(n).g(4).test(pytorch_qnnp_x8zip_x4__sse2);
          }

        */
    }

    #[test] fn X8ZIP_X4__SSE2_n_gt_16() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 17; n < 32; n++) {
            ZipMicrokernelTester().n(n).g(4).test(pytorch_qnnp_x8zip_x4__sse2);
          }

        */
    }

    #[test] fn X8ZIP_X4__SSE2_n_lt_16() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 16; n++) {
            ZipMicrokernelTester().n(n).g(4).test(pytorch_qnnp_x8zip_x4__sse2);
          }

        */
    }

    #[test] fn X8ZIP_XM__SSE2_n_eq_8_m_4() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          ZipMicrokernelTester().n(8).g(4).test(pytorch_qnnp_x8zip_xm__sse2);

        */
    }

    #[test] fn X8ZIP_XM__SSE2_n_eq_8_m_div_4() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize g = 4; g < 32; g += 4) {
            ZipMicrokernelTester().n(8).g(g).test(pytorch_qnnp_x8zip_xm__sse2);
          }

        */
    }

    #[test] fn X8ZIP_XM__SSE2_n_eq_8_m_gt_4() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize g = 5; g < 8; g++) {
            ZipMicrokernelTester().n(8).g(g).test(pytorch_qnnp_x8zip_xm__sse2);
          }

        */
    }

    #[test] fn X8ZIP_XM__SSE2_n_eq_16_m_4() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          ZipMicrokernelTester().n(16).g(4).test(pytorch_qnnp_x8zip_xm__sse2);

        */
    }

    #[test] fn X8ZIP_XM__SSE2_n_eq_16_m_div_4() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize g = 4; g < 32; g += 4) {
            ZipMicrokernelTester().n(16).g(g).test(pytorch_qnnp_x8zip_xm__sse2);
          }

        */
    }

    #[test] fn X8ZIP_XM__SSE2_n_eq_16_m_gt_4() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize g = 5; g < 8; g++) {
            ZipMicrokernelTester().n(16).g(g).test(pytorch_qnnp_x8zip_xm__sse2);
          }

        */
    }

    #[test] fn X8ZIP_XM__SSE2_n_div_16_m_eq_4() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 16; n < 256; n += 16) {
            ZipMicrokernelTester().n(n).g(4).test(pytorch_qnnp_x8zip_xm__sse2);
          }

        */
    }

    #[test] fn X8ZIP_XM__SSE2_n_div_16_m_4() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 16; n < 256; n += 16) {
            for (usize g = 4; g < 32; g += 4) {
              ZipMicrokernelTester().n(n).g(g).test(pytorch_qnnp_x8zip_xm__sse2);
            }
          }

        */
    }

    #[test] fn X8ZIP_XM__SSE2_n_div_16_m_gt_4() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 16; n < 256; n += 16) {
            for (usize g = 5; g < 8; g++) {
              ZipMicrokernelTester().n(n).g(g).test(pytorch_qnnp_x8zip_xm__sse2);
            }
          }

        */
    }

    #[test] fn X8ZIP_XM__SSE2_n_gt_16_m_eq_4() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 17; n < 32; n++) {
            ZipMicrokernelTester().n(n).g(4).test(pytorch_qnnp_x8zip_xm__sse2);
          }

        */
    }

    #[test] fn X8ZIP_XM__SSE2_n_gt_16_m_div_4() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 17; n < 32; n++) {
            for (usize g = 4; g < 32; g += 4) {
              ZipMicrokernelTester().n(n).g(g).test(pytorch_qnnp_x8zip_xm__sse2);
            }
          }

        */
    }

    #[test] fn X8ZIP_XM__SSE2_n_gt_16_m_4() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 17; n < 32; n++) {
            for (usize g = 5; g < 8; g++) {
              ZipMicrokernelTester().n(n).g(g).test(pytorch_qnnp_x8zip_xm__sse2);
            }
          }

        */
    }

    #[test] fn X8ZIP_XM__SSE2_n_lt_16() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize n = 1; n < 16; n++) {
            for (usize g = 4; g < 12; g++) {
              ZipMicrokernelTester().n(n).g(g).test(pytorch_qnnp_x8zip_xm__sse2);
            }
          }

        */
    }
}
