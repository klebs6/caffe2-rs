// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/q8conv.cc]

#[cfg(CPUINFO_ARCH_ARM)]
mod arch_arm {
    use super::*;

    #[test] fn q8conv_4x8_aarch32_neon_k_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(4)
              .nr(8)
              .np(8)
              .kr(1)
              .m(4)
              .n(8)
              .k(8)
              .aStride(37)
              .test(pytorch_q8conv_ukernel_4x8__aarch32_neon);

        */
    }

    #[test] fn q8conv_4x8_aarch32_neon_k_eq_8_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(4)
              .nr(8)
              .np(8)
              .kr(1)
              .m(4)
              .n(8)
              .k(8)
              .aStride(37)
              .cStride(17)
              .test(pytorch_q8conv_ukernel_4x8__aarch32_neon);

        */
    }

    #[test] fn q8conv_4x8_aarch32_neon_k_eq_8_qmin128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmin(128).test(
              pytorch_q8conv_ukernel_4x8__aarch32_neon);

        */
    }

    #[test] fn q8conv_4x8_aarch32_neon_k_eq_8_qmax128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmax(128).test(
              pytorch_q8conv_ukernel_4x8__aarch32_neon);

        */
    }

    #[test] fn q8conv_4x8_aarch32_neon_k_eq_8_azp_only() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(4)
              .nr(8)
              .np(8)
              .kr(1)
              .m(4)
              .n(8)
              .k(8)
              .aZeroPoint(255)
              .bZeroPoint(0)
              .test(pytorch_q8conv_ukernel_4x8__aarch32_neon);

        */
    }

    #[test] fn q8conv_4x8_aarch32_neon_k_eq_8_bzp_only() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(4)
              .nr(8)
              .np(8)
              .kr(1)
              .m(4)
              .n(8)
              .k(8)
              .aZeroPoint(0)
              .bZeroPoint(255)
              .test(pytorch_q8conv_ukernel_4x8__aarch32_neon);

        */
    }

    #[test] fn q8conv_4x8_aarch32_neon_k_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(8)
                .np(8)
                .kr(1)
                .m(4)
                .n(8)
                .k(k)
                .aStride(37)
                .test(pytorch_q8conv_ukernel_4x8__aarch32_neon);
          }

        */
    }

    #[test] fn q8conv_4x8_aarch32_neon_k_gt_8_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(8)
                .np(8)
                .kr(1)
                .m(4)
                .n(8)
                .k(k)
                .aStride(37)
                .cStride(17)
                .test(pytorch_q8conv_ukernel_4x8__aarch32_neon);
          }

        */
    }

    #[test] fn q8conv_4x8_aarch32_neon_k_gt_8_azp_only() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(8)
                .np(8)
                .kr(1)
                .m(4)
                .n(8)
                .k(k)
                .aStride(37)
                .aZeroPoint(255)
                .bZeroPoint(0)
                .test(pytorch_q8conv_ukernel_4x8__aarch32_neon);
          }

        */
    }

    #[test] fn q8conv_4x8_aarch32_neon_k_gt_8_bzp_only() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(8)
                .np(8)
                .kr(1)
                .m(4)
                .n(8)
                .k(k)
                .aStride(37)
                .aZeroPoint(0)
                .bZeroPoint(255)
                .test(pytorch_q8conv_ukernel_4x8__aarch32_neon);
          }

        */
    }

    #[test] fn q8conv_4x8_aarch32_neon_k_gt_8_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            for (u32 m = 1; m <= 4; m++) {
              for (u32 n = 1; n <= 8; n++) {
                GemmMicrokernelTester()
                    .mr(4)
                    .nr(8)
                    .np(8)
                    .kr(1)
                    .m(m)
                    .n(n)
                    .k(k)
                    .aStride(37)
                    .iterations(3)
                    .test(pytorch_q8conv_ukernel_4x8__aarch32_neon);
              }
            }
          }

        */
    }

    #[test] fn q8conv_4x8_aarch32_neon_k_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(8)
                .np(8)
                .kr(1)
                .m(4)
                .n(8)
                .k(k)
                .aStride(171)
                .test(pytorch_q8conv_ukernel_4x8__aarch32_neon);
          }

        */
    }

    #[test] fn q8conv_4x8_aarch32_neon_k_div_8_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(8)
                .np(8)
                .kr(1)
                .m(4)
                .n(8)
                .k(k)
                .aStride(171)
                .cStride(17)
                .test(pytorch_q8conv_ukernel_4x8__aarch32_neon);
          }

        */
    }

    #[test] fn q8conv_4x8_aarch32_neon_k_div_8_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 16; k < 128; k += 24) {
            for (u32 m = 1; m <= 4; m++) {
              for (u32 n = 1; n <= 8; n++) {
                GemmMicrokernelTester()
                    .mr(4)
                    .nr(8)
                    .np(8)
                    .kr(1)
                    .m(m)
                    .n(n)
                    .k(k)
                    .aStride(171)
                    .iterations(3)
                    .test(pytorch_q8conv_ukernel_4x8__aarch32_neon);
              }
            }
          }

        */
    }
}

#[cfg(CPUINFO_ARCH_ARM64)]
mod arch_arm64 {
    use super::*;

    #[test] fn q8conv_8x8_aarch64_neon_k_eq_8() {
        todo!();
        /*
        
          GemmMicrokernelTester()
              .mr(8)
              .nr(8)
              .np(8)
              .kr(1)
              .m(8)
              .n(8)
              .k(8)
              .aStride(37)
              .test(pytorch_q8conv_ukernel_8x8__aarch64_neon);

        */
    }

    #[test] fn q8conv_8x8_aarch64_neon_k_eq_8_strided_c() {
        todo!();
        /*
        
          GemmMicrokernelTester()
              .mr(8)
              .nr(8)
              .np(8)
              .kr(1)
              .m(8)
              .n(8)
              .k(8)
              .aStride(37)
              .cStride(17)
              .test(pytorch_q8conv_ukernel_8x8__aarch64_neon);

        */
    }

    #[test] fn q8conv_8x8_aarch64_neon_k_eq_8_qmin128() {
        todo!();
        /*
        
          GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).qmin(128).test(
              pytorch_q8conv_ukernel_8x8__aarch64_neon);

        */
    }

    #[test] fn q8conv_8x8_aarch64_neon_k_eq_8_qmax128() {
        todo!();
        /*
        
          GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).qmax(128).test(
              pytorch_q8conv_ukernel_8x8__aarch64_neon);

        */
    }

    #[test] fn q8conv_8x8_aarch64_neon_k_eq_8_azp_only() {
        todo!();
        /*
        
          GemmMicrokernelTester()
              .mr(8)
              .nr(8)
              .np(8)
              .kr(1)
              .m(8)
              .n(8)
              .k(8)
              .aZeroPoint(255)
              .bZeroPoint(0)
              .test(pytorch_q8conv_ukernel_8x8__aarch64_neon);

        */
    }

    #[test] fn q8conv_8x8_aarch64_neon_k_eq_8_bzp_only() {
        todo!();
        /*
        
          GemmMicrokernelTester()
              .mr(8)
              .nr(8)
              .np(8)
              .kr(1)
              .m(8)
              .n(8)
              .k(8)
              .aZeroPoint(0)
              .bZeroPoint(255)
              .test(pytorch_q8conv_ukernel_8x8__aarch64_neon);

        */
    }

    #[test] fn q8conv_8x8_aarch64_neon_k_gt_8() {
        todo!();
        /*
        
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(8)
                .nr(8)
                .np(8)
                .kr(1)
                .m(8)
                .n(8)
                .k(k)
                .aStride(37)
                .test(pytorch_q8conv_ukernel_8x8__aarch64_neon);
          }

        */
    }

    #[test] fn q8conv_8x8_aarch64_neon_k_gt_8_strided_c() {
        todo!();
        /*
        
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(8)
                .nr(8)
                .np(8)
                .kr(1)
                .m(8)
                .n(8)
                .k(k)
                .aStride(37)
                .cStride(17)
                .test(pytorch_q8conv_ukernel_8x8__aarch64_neon);
          }

        */
    }

    #[test] fn q8conv_8x8_aarch64_neon_k_gt_8_azp_only() {
        todo!();
        /*
        
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(8)
                .nr(8)
                .np(8)
                .kr(1)
                .m(8)
                .n(8)
                .k(k)
                .aStride(37)
                .aZeroPoint(255)
                .bZeroPoint(0)
                .test(pytorch_q8conv_ukernel_8x8__aarch64_neon);
          }

        */
    }

    #[test] fn q8conv_8x8_aarch64_neon_k_gt_8_bzp_only() {
        todo!();
        /*
        
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(8)
                .nr(8)
                .np(8)
                .kr(1)
                .m(8)
                .n(8)
                .k(k)
                .aStride(37)
                .aZeroPoint(0)
                .bZeroPoint(255)
                .test(pytorch_q8conv_ukernel_8x8__aarch64_neon);
          }

        */
    }

    #[test] fn q8conv_8x8_aarch64_neon_k_gt_8_subtile() {
        todo!();
        /*
        
          for (usize k = 9; k < 16; k++) {
            for (u32 m = 1; m <= 8; m++) {
              for (u32 n = 1; n <= 8; n++) {
                GemmMicrokernelTester()
                    .mr(8)
                    .nr(8)
                    .np(8)
                    .kr(1)
                    .m(m)
                    .n(n)
                    .k(k)
                    .aStride(37)
                    .iterations(3)
                    .test(pytorch_q8conv_ukernel_8x8__aarch64_neon);
              }
            }
          }

        */
    }

    #[test] fn q8conv_8x8_aarch64_neon_k_div_8() {
        todo!();
        /*
        
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester()
                .mr(8)
                .nr(8)
                .np(8)
                .kr(1)
                .m(8)
                .n(8)
                .k(k)
                .aStride(171)
                .test(pytorch_q8conv_ukernel_8x8__aarch64_neon);
          }

        */
    }

    #[test] fn q8conv_8x8_aarch64_neon_k_div_8_strided_c() {
        todo!();
        /*
        
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester()
                .mr(8)
                .nr(8)
                .np(8)
                .kr(1)
                .m(8)
                .n(8)
                .k(k)
                .aStride(171)
                .cStride(17)
                .test(pytorch_q8conv_ukernel_8x8__aarch64_neon);
          }

        */
    }

    #[test] fn q8conv_8x8_aarch64_neon_k_div_8_subtile() {
        todo!();
        /*
        
          for (usize k = 16; k < 128; k += 24) {
            for (u32 m = 1; m <= 8; m++) {
              for (u32 n = 1; n <= 8; n++) {
                GemmMicrokernelTester()
                    .mr(8)
                    .nr(8)
                    .np(8)
                    .kr(1)
                    .m(m)
                    .n(n)
                    .k(k)
                    .aStride(171)
                    .iterations(3)
                    .test(pytorch_q8conv_ukernel_8x8__aarch64_neon);
              }
            }
          }

        */
    }
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod arch_arm {

    use super::*;

    #[test] fn q8conv_4x8_neon_k_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(4)
              .nr(8)
              .np(8)
              .kr(1)
              .m(4)
              .n(8)
              .k(8)
              .aStride(37)
              .test(pytorch_q8conv_ukernel_4x8__neon);

        */
    }

    #[test] fn q8conv_4x8_neon_k_eq_8_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(4)
              .nr(8)
              .np(8)
              .kr(1)
              .m(4)
              .n(8)
              .k(8)
              .aStride(37)
              .cStride(17)
              .test(pytorch_q8conv_ukernel_4x8__neon);

        */
    }

    #[test] fn q8conv_4x8_neon_k_eq_8_qmin128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmin(128).test(
              pytorch_q8conv_ukernel_4x8__neon);

        */
    }

    #[test] fn q8conv_4x8_neon_k_eq_8_qmax128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmax(128).test(
              pytorch_q8conv_ukernel_4x8__neon);

        */
    }

    #[test] fn q8conv_4x8_neon_k_eq_8_azp_only() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(4)
              .nr(8)
              .np(8)
              .kr(1)
              .m(4)
              .n(8)
              .k(8)
              .aZeroPoint(255)
              .bZeroPoint(0)
              .test(pytorch_q8conv_ukernel_4x8__neon);

        */
    }

    #[test] fn q8conv_4x8_neon_k_eq_8_bzp_only() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(4)
              .nr(8)
              .np(8)
              .kr(1)
              .m(4)
              .n(8)
              .k(8)
              .aZeroPoint(0)
              .bZeroPoint(255)
              .test(pytorch_q8conv_ukernel_4x8__neon);

        */
    }

    #[test] fn q8conv_4x8_neon_k_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(8)
                .np(8)
                .kr(1)
                .m(4)
                .n(8)
                .k(k)
                .aStride(37)
                .test(pytorch_q8conv_ukernel_4x8__neon);
          }

        */
    }

    #[test] fn q8conv_4x8_neon_k_gt_8_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(8)
                .np(8)
                .kr(1)
                .m(4)
                .n(8)
                .k(k)
                .aStride(37)
                .cStride(17)
                .test(pytorch_q8conv_ukernel_4x8__neon);
          }

        */
    }

    #[test] fn q8conv_4x8_neon_k_gt_8_azp_only() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(8)
                .np(8)
                .kr(1)
                .m(4)
                .n(8)
                .k(k)
                .aStride(37)
                .aZeroPoint(255)
                .bZeroPoint(0)
                .test(pytorch_q8conv_ukernel_4x8__neon);
          }

        */
    }

    #[test] fn q8conv_4x8_neon_k_gt_8_bzp_only() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(8)
                .np(8)
                .kr(1)
                .m(4)
                .n(8)
                .k(k)
                .aStride(37)
                .aZeroPoint(0)
                .bZeroPoint(255)
                .test(pytorch_q8conv_ukernel_4x8__neon);
          }

        */
    }

    #[test] fn q8conv_4x8_neon_k_gt_8_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            for (u32 m = 1; m <= 4; m++) {
              for (u32 n = 1; n <= 8; n++) {
                GemmMicrokernelTester()
                    .mr(4)
                    .nr(8)
                    .np(8)
                    .kr(1)
                    .m(m)
                    .n(n)
                    .k(k)
                    .aStride(37)
                    .iterations(3)
                    .test(pytorch_q8conv_ukernel_4x8__neon);
              }
            }
          }

        */
    }

    #[test] fn q8conv_4x8_neon_k_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(8)
                .np(8)
                .kr(1)
                .m(4)
                .n(8)
                .k(k)
                .aStride(171)
                .test(pytorch_q8conv_ukernel_4x8__neon);
          }

        */
    }

    #[test] fn q8conv_4x8_neon_k_div_8_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(8)
                .np(8)
                .kr(1)
                .m(4)
                .n(8)
                .k(k)
                .aStride(171)
                .cStride(17)
                .test(pytorch_q8conv_ukernel_4x8__neon);
          }

        */
    }

    #[test] fn q8conv_4x8_neon_k_div_8_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 16; k < 128; k += 24) {
            for (u32 m = 1; m <= 4; m++) {
              for (u32 n = 1; n <= 8; n++) {
                GemmMicrokernelTester()
                    .mr(4)
                    .nr(8)
                    .np(8)
                    .kr(1)
                    .m(m)
                    .n(n)
                    .k(k)
                    .aStride(171)
                    .iterations(3)
                    .test(pytorch_q8conv_ukernel_4x8__neon);
              }
            }
          }

        */
    }

    #[test] fn q8conv_8x8_neon_k_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(8)
              .nr(8)
              .np(8)
              .kr(1)
              .m(8)
              .n(8)
              .k(8)
              .aStride(37)
              .test(pytorch_q8conv_ukernel_8x8__neon);

        */
    }

    #[test] fn q8conv_8x8_neon_k_eq_8_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(8)
              .nr(8)
              .np(8)
              .kr(1)
              .m(8)
              .n(8)
              .k(8)
              .aStride(37)
              .cStride(17)
              .test(pytorch_q8conv_ukernel_8x8__neon);

        */
    }

    #[test] fn q8conv_8x8_neon_k_eq_8_qmin128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).qmin(128).test(
              pytorch_q8conv_ukernel_8x8__neon);

        */
    }

    #[test] fn q8conv_8x8_neon_k_eq_8_qmax128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).qmax(128).test(
              pytorch_q8conv_ukernel_8x8__neon);

        */
    }

    #[test] fn q8conv_8x8_neon_k_eq_8_azp_only() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(8)
              .nr(8)
              .np(8)
              .kr(1)
              .m(8)
              .n(8)
              .k(8)
              .aZeroPoint(255)
              .bZeroPoint(0)
              .test(pytorch_q8conv_ukernel_8x8__neon);

        */
    }

    #[test] fn q8conv_8x8_neon_k_eq_8_bzp_only() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(8)
              .nr(8)
              .np(8)
              .kr(1)
              .m(8)
              .n(8)
              .k(8)
              .aZeroPoint(0)
              .bZeroPoint(255)
              .test(pytorch_q8conv_ukernel_8x8__neon);

        */
    }

    #[test] fn q8conv_8x8_neon_k_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(8)
                .nr(8)
                .np(8)
                .kr(1)
                .m(8)
                .n(8)
                .k(k)
                .aStride(37)
                .test(pytorch_q8conv_ukernel_8x8__neon);
          }

        */
    }

    #[test] fn q8conv_8x8_neon_k_gt_8_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(8)
                .nr(8)
                .np(8)
                .kr(1)
                .m(8)
                .n(8)
                .k(k)
                .aStride(37)
                .cStride(17)
                .test(pytorch_q8conv_ukernel_8x8__neon);
          }

        */
    }

    #[test] fn q8conv_8x8_neon_k_gt_8_azp_only() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(8)
                .nr(8)
                .np(8)
                .kr(1)
                .m(8)
                .n(8)
                .k(k)
                .aStride(37)
                .aZeroPoint(255)
                .bZeroPoint(0)
                .test(pytorch_q8conv_ukernel_8x8__neon);
          }

        */
    }

    #[test] fn q8conv_8x8_neon_k_gt_8_bzp_only() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(8)
                .nr(8)
                .np(8)
                .kr(1)
                .m(8)
                .n(8)
                .k(k)
                .aStride(37)
                .aZeroPoint(0)
                .bZeroPoint(255)
                .test(pytorch_q8conv_ukernel_8x8__neon);
          }

        */
    }

    #[test] fn q8conv_8x8_neon_k_gt_8_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            for (u32 m = 1; m <= 8; m++) {
              for (u32 n = 1; n <= 8; n++) {
                GemmMicrokernelTester()
                    .mr(8)
                    .nr(8)
                    .np(8)
                    .kr(1)
                    .m(m)
                    .n(n)
                    .k(k)
                    .aStride(37)
                    .iterations(3)
                    .test(pytorch_q8conv_ukernel_8x8__neon);
              }
            }
          }

        */
    }

    #[test] fn q8conv_8x8_neon_k_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester()
                .mr(8)
                .nr(8)
                .np(8)
                .kr(1)
                .m(8)
                .n(8)
                .k(k)
                .aStride(171)
                .test(pytorch_q8conv_ukernel_8x8__neon);
          }

        */
    }

    #[test] fn q8conv_8x8_neon_k_div_8_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester()
                .mr(8)
                .nr(8)
                .np(8)
                .kr(1)
                .m(8)
                .n(8)
                .k(k)
                .aStride(171)
                .cStride(17)
                .test(pytorch_q8conv_ukernel_8x8__neon);
          }

        */
    }

    #[test] fn q8conv_8x8_neon_k_div_8_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 16; k < 128; k += 24) {
            for (u32 m = 1; m <= 8; m++) {
              for (u32 n = 1; n <= 8; n++) {
                GemmMicrokernelTester()
                    .mr(8)
                    .nr(8)
                    .np(8)
                    .kr(1)
                    .m(m)
                    .n(n)
                    .k(k)
                    .aStride(171)
                    .iterations(3)
                    .test(pytorch_q8conv_ukernel_8x8__neon);
              }
            }
          }

        */
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {

    use super::*;

    #[test] fn q8conv_4x4c2_sse2_k_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester()
              .mr(4)
              .nr(4)
              .np(4)
              .kr(2)
              .m(4)
              .n(4)
              .k(8)
              .aStride(37)
              .test(pytorch_q8conv_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8conv_4x4c2_sse2_k_eq_8_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester()
              .mr(4)
              .nr(4)
              .np(4)
              .kr(2)
              .m(4)
              .n(4)
              .k(8)
              .aStride(37)
              .cStride(17)
              .test(pytorch_q8conv_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8conv_4x4c2_sse2_k_eq_8_qmin128() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(8).qmin(128).test(
              pytorch_q8conv_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8conv_4x4c2_sse2_k_eq_8_qmax128() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(8).qmax(128).test(
              pytorch_q8conv_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8conv_4x4c2_sse2_k_eq_8_azp_only() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester()
              .mr(4)
              .nr(4)
              .np(4)
              .kr(2)
              .m(4)
              .n(4)
              .k(8)
              .aZeroPoint(255)
              .bZeroPoint(0)
              .test(pytorch_q8conv_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8conv_4x4c2_sse2_k_eq_8_bzp_only() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester()
              .mr(4)
              .nr(4)
              .np(4)
              .kr(2)
              .m(4)
              .n(4)
              .k(8)
              .aZeroPoint(0)
              .bZeroPoint(255)
              .test(pytorch_q8conv_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8conv_4x4c2_sse2_k_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(4)
                .np(4)
                .kr(2)
                .m(4)
                .n(4)
                .k(k)
                .aStride(37)
                .test(pytorch_q8conv_ukernel_4x4c2__sse2);
          }

        */
    }

    #[test] fn q8conv_4x4c2_sse2_k_gt_8_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(4)
                .np(4)
                .kr(2)
                .m(4)
                .n(4)
                .k(k)
                .aStride(37)
                .cStride(17)
                .test(pytorch_q8conv_ukernel_4x4c2__sse2);
          }

        */
    }

    #[test] fn q8conv_4x4c2_sse2_k_gt_8_azp_only() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(4)
                .np(4)
                .kr(2)
                .m(4)
                .n(4)
                .k(k)
                .aStride(37)
                .aZeroPoint(255)
                .bZeroPoint(0)
                .test(pytorch_q8conv_ukernel_4x4c2__sse2);
          }

        */
    }

    #[test] fn q8conv_4x4c2_sse2_k_gt_8_bzp_only() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(4)
                .np(4)
                .kr(2)
                .m(4)
                .n(4)
                .k(k)
                .aStride(37)
                .aZeroPoint(0)
                .bZeroPoint(255)
                .test(pytorch_q8conv_ukernel_4x4c2__sse2);
          }

        */
    }

    #[test] fn q8conv_4x4c2_sse2_k_gt_8_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize k = 9; k < 16; k++) {
            for (u32 m = 1; m <= 4; m++) {
              for (u32 n = 1; n <= 4; n++) {
                GemmMicrokernelTester()
                    .mr(4)
                    .nr(4)
                    .np(4)
                    .kr(2)
                    .m(m)
                    .n(n)
                    .k(k)
                    .aStride(37)
                    .iterations(3)
                    .test(pytorch_q8conv_ukernel_4x4c2__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8conv_4x4c2_sse2_k_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(4)
                .np(4)
                .kr(2)
                .m(4)
                .n(4)
                .k(k)
                .aStride(171)
                .test(pytorch_q8conv_ukernel_4x4c2__sse2);
          }

        */
    }

    #[test] fn q8conv_4x4c2_sse2_k_div_8_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(4)
                .np(4)
                .kr(2)
                .m(4)
                .n(4)
                .k(k)
                .aStride(171)
                .cStride(17)
                .test(pytorch_q8conv_ukernel_4x4c2__sse2);
          }

        */
    }

    #[test] fn q8conv_4x4c2_sse2_k_div_8_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize k = 16; k < 128; k += 24) {
            for (u32 m = 1; m <= 4; m++) {
              for (u32 n = 1; n <= 4; n++) {
                GemmMicrokernelTester()
                    .mr(4)
                    .nr(4)
                    .np(4)
                    .kr(2)
                    .m(m)
                    .n(n)
                    .k(k)
                    .aStride(171)
                    .iterations(3)
                    .test(pytorch_q8conv_ukernel_4x4c2__sse2);
              }
            }
          }

        */
    }
}
