// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/q8gemm.cc]

#[cfg(CPUINFO_ARCH_ARM)]
mod arm {
    use super::*;

    #[test] fn q8gemm_4x8_aarch32_neon_k_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).test(
              pytorch_q8gemm_ukernel_4x8__aarch32_neon);

        */
    }

    #[test] fn q8gemm_4x8_aarch32_neon_k_eq_8_strided_a() {
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
              .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);

        */
    }

    #[test] fn q8gemm_4x8_aarch32_neon_k_eq_8_strided_c() {
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
              .cStride(17)
              .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);

        */
    }

    #[test] fn q8gemm_4x8_aarch32_neon_k_eq_8_qmin128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmin(128).test(
              pytorch_q8gemm_ukernel_4x8__aarch32_neon);

        */
    }

    #[test] fn q8gemm_4x8_aarch32_neon_k_eq_8_qmax128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmax(128).test(
              pytorch_q8gemm_ukernel_4x8__aarch32_neon);

        */
    }

    #[test] fn q8gemm_4x8_aarch32_neon_k_eq_8_azp0() {
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
              .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);

        */
    }

    #[test] fn q8gemm_4x8_aarch32_neon_k_eq_8_bzp0() {
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
              .bZeroPoint(0)
              .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);

        */
    }

    #[test] fn q8gemm_4x8_aarch32_neon_k_eq_8_nozp() {
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
              .bZeroPoint(0)
              .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);

        */
    }

    #[test] fn q8gemm_4x8_aarch32_neon_k_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(k).test(
                pytorch_q8gemm_ukernel_4x8__aarch32_neon);
          }

        */
    }

    #[test] fn q8gemm_4x8_aarch32_neon_k_gt_8_strided_a() {
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
                .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
          }

        */
    }

    #[test] fn q8gemm_4x8_aarch32_neon_k_gt_8_strided_c() {
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
                .cStride(17)
                .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
          }

        */
    }

    #[test] fn q8gemm_4x8_aarch32_neon_k_gt_8_azp0() {
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
                .aZeroPoint(0)
                .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
          }

        */
    }

    #[test] fn q8gemm_4x8_aarch32_neon_k_gt_8_bzp0() {
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
                .bZeroPoint(0)
                .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
          }

        */
    }

    #[test] fn q8gemm_4x8_aarch32_neon_k_gt_8_nozp() {
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
                .aZeroPoint(0)
                .bZeroPoint(0)
                .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
          }

        */
    }

    #[test] fn q8gemm_4x8_aarch32_neon_k_gt_8_subtile() {
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
                    .iterations(3)
                    .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
              }
            }
          }

        */
    }

    #[test] fn q8gemm_4x8_aarch32_neon_k_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(k).test(
                pytorch_q8gemm_ukernel_4x8__aarch32_neon);
          }

        */
    }

    #[test] fn q8gemm_4x8_aarch32_neon_k_div_8_strided_a() {
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
                .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
          }

        */
    }

    #[test] fn q8gemm_4x8_aarch32_neon_k_div_8_strided_c() {
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
                .cStride(17)
                .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
          }

        */
    }

    #[test] fn q8gemm_4x8_aarch32_neon_k_div_8_subtile() {
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
                    .iterations(3)
                    .test(pytorch_q8gemm_ukernel_4x8__aarch32_neon);
              }
            }
          }

        */
    }

    //
    // Dynamic Quantization
    //

    #[test] fn q8gemm_dq_4x8_aarch32_neon_k_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).test(
              pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);

        */
    }

    #[test] fn q8gemm_dq_4x8_aarch32_neon_k_eq_8_strided_a() {
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
              .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);

        */
    }

    #[test] fn q8gemm_dq_4x8_aarch32_neon_k_eq_8_strided_c() {
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
              .cStride(17)
              .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);

        */
    }

    #[test] fn q8gemm_dq_4x8_aarch32_neon_k_eq_8_qmin128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmin(128).test(
              pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);

        */
    }

    #[test] fn q8gemm_dq_4x8_aarch32_neon_k_eq_8_qmax128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmax(128).test(
              pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);

        */
    }

    #[test] fn q8gemm_dq_4x8_aarch32_neon_k_eq_8_azp0() {
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
              .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);

        */
    }

    #[test] fn q8gemm_dq_4x8_aarch32_neon_k_eq_8_bzp0() {
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
              .bZeroPoint(0)
              .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);

        */
    }

    #[test] fn q8gemm_dq_4x8_aarch32_neon_k_eq_8_nozp() {
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
              .bZeroPoint(0)
              .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);

        */
    }

    #[test] fn q8gemm_dq_4x8_aarch32_neon_k_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(k).test(
                pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
          }

        */
    }

    #[test] fn q8gemm_dq_4x8_aarch32_neon_k_gt_8_strided_a() {
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
                .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
          }

        */
    }

    #[test] fn q8gemm_dq_4x8_aarch32_neon_k_gt_8_strided_c() {
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
                .cStride(17)
                .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
          }

        */
    }

    #[test] fn q8gemm_dq_4x8_aarch32_neon_k_gt_8_azp0() {
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
                .aZeroPoint(0)
                .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
          }

        */
    }

    #[test] fn q8gemm_dq_4x8_aarch32_neon_k_gt_8_bzp0() {
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
                .bZeroPoint(0)
                .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
          }

        */
    }

    #[test] fn q8gemm_dq_4x8_aarch32_neon_k_gt_8_nozp() {
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
                .aZeroPoint(0)
                .bZeroPoint(0)
                .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
          }

        */
    }

    #[test] fn q8gemm_dq_4x8_aarch32_neon_k_gt_8_subtile() {
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
                    .iterations(3)
                    .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
              }
            }
          }

        */
    }

    #[test] fn q8gemm_dq_4x8_aarch32_neon_k_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(k).test(
                pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
          }

        */
    }

    #[test] fn q8gemm_dq_4x8_aarch32_neon_k_div_8_strided_a() {
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
                .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
          }

        */
    }

    #[test] fn q8gemm_dq_4x8_aarch32_neon_k_div_8_strided_c() {
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
                .cStride(17)
                .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
          }

        */
    }

    #[test] fn q8gemm_dq_4x8_aarch32_neon_k_div_8_subtile() {
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
                    .iterations(3)
                    .test(pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon);
              }
            }
          }

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_aarch32_neon_k_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(4).nr(8).np(8).kr(2).m(4).n(8).k(8).test(
              pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_aarch32_neon_k_eq_8_strided_a() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(4)
              .nr(8)
              .np(8)
              .kr(2)
              .m(4)
              .n(8)
              .k(8)
              .aStride(37)
              .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_aarch32_neon_k_eq_8_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(4)
              .nr(8)
              .np(8)
              .kr(2)
              .m(4)
              .n(8)
              .k(8)
              .cStride(17)
              .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_aarch32_neon_k_eq_8_qmin128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(4).nr(8).np(8).kr(2).m(4).n(8).k(8).qmin(128).test(
              pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_aarch32_neon_k_eq_8_qmax128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(4).nr(8).np(8).kr(2).m(4).n(8).k(8).qmax(128).test(
              pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_aarch32_neon_k_eq_8_azp0() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(4)
              .nr(8)
              .np(8)
              .kr(2)
              .m(4)
              .n(8)
              .k(8)
              .aZeroPoint(0)
              .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_aarch32_neon_k_eq_8_bzp0() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(4)
              .nr(8)
              .np(8)
              .kr(2)
              .m(4)
              .n(8)
              .k(8)
              .bZeroPoint(0)
              .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_aarch32_neon_k_eq_8_nozp() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(4)
              .nr(8)
              .np(8)
              .kr(2)
              .m(4)
              .n(8)
              .k(8)
              .aZeroPoint(0)
              .bZeroPoint(0)
              .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_aarch32_neon_k_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester().mr(4).nr(8).np(8).kr(2).m(4).n(8).k(k).test(
                pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
          }

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_aarch32_neon_k_gt_8_strided_a() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(8)
                .np(8)
                .kr(2)
                .m(4)
                .n(8)
                .k(k)
                .aStride(37)
                .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
          }

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_aarch32_neon_k_gt_8_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(8)
                .np(8)
                .kr(2)
                .m(4)
                .n(8)
                .k(k)
                .cStride(17)
                .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
          }

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_aarch32_neon_k_gt_8_azp0() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(8)
                .np(8)
                .kr(2)
                .m(4)
                .n(8)
                .k(k)
                .aZeroPoint(0)
                .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
          }

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_aarch32_neon_k_gt_8_bzp0() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(8)
                .np(8)
                .kr(2)
                .m(4)
                .n(8)
                .k(k)
                .bZeroPoint(0)
                .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
          }

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_aarch32_neon_k_gt_8_nozp() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(8)
                .np(8)
                .kr(2)
                .m(4)
                .n(8)
                .k(k)
                .aZeroPoint(0)
                .bZeroPoint(0)
                .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
          }

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_aarch32_neon_k_gt_8_subtile() {
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
                    .kr(2)
                    .m(m)
                    .n(n)
                    .k(k)
                    .iterations(3)
                    .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
              }
            }
          }

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_aarch32_neon_k_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester().mr(4).nr(8).np(8).kr(2).m(4).n(8).k(k).test(
                pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
          }

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_aarch32_neon_k_div_8_strided_a() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(8)
                .np(8)
                .kr(2)
                .m(4)
                .n(8)
                .k(k)
                .aStride(171)
                .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
          }

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_aarch32_neon_k_div_8_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(8)
                .np(8)
                .kr(2)
                .m(4)
                .n(8)
                .k(k)
                .cStride(17)
                .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
          }

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_aarch32_neon_k_div_8_subtile() {
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
                    .kr(2)
                    .m(m)
                    .n(n)
                    .k(k)
                    .iterations(3)
                    .test(pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon);
              }
            }
          }

        */
    }

}

#[cfg(CPUINFO_ARCH_ARM64)]
mod arm64 {
    use super::*;

    #[test] fn q8gemm_8x8_aarch64_neon_k_eq_8() {
        todo!();
        /*
        
          GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).test(
              pytorch_q8gemm_ukernel_8x8__aarch64_neon);

        */
    }

    #[test] fn q8gemm_8x8_aarch64_neon_k_eq_8_strided_a() {
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
              .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);

        */
    }

    #[test] fn q8gemm_8x8_aarch64_neon_k_eq_8_strided_c() {
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
              .cStride(17)
              .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);

        */
    }

    #[test] fn q8gemm_8x8_aarch64_neon_k_eq_8_qmin128() {
        todo!();
        /*
        
          GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).qmin(128).test(
              pytorch_q8gemm_ukernel_8x8__aarch64_neon);

        */
    }

    #[test] fn q8gemm_8x8_aarch64_neon_k_eq_8_qmax128() {
        todo!();
        /*
        
          GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).qmax(128).test(
              pytorch_q8gemm_ukernel_8x8__aarch64_neon);

        */
    }

    #[test] fn q8gemm_8x8_aarch64_neon_k_eq_8_azp0() {
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
              .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);

        */
    }

    #[test] fn q8gemm_8x8_aarch64_neon_k_eq_8_bzp0() {
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
              .bZeroPoint(0)
              .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);

        */
    }

    #[test] fn q8gemm_8x8_aarch64_neon_k_eq_8_nozp() {
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
              .bZeroPoint(0)
              .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);

        */
    }

    #[test] fn q8gemm_8x8_aarch64_neon_k_gt_8() {
        todo!();
        /*
        
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(k).test(
                pytorch_q8gemm_ukernel_8x8__aarch64_neon);
          }

        */
    }

    #[test] fn q8gemm_8x8_aarch64_neon_k_gt_8_strided_a() {
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
                .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
          }

        */
    }

    #[test] fn q8gemm_8x8_aarch64_neon_k_gt_8_strided_c() {
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
                .cStride(17)
                .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
          }

        */
    }

    #[test] fn q8gemm_8x8_aarch64_neon_k_gt_8_azp0() {
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
                .aZeroPoint(0)
                .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
          }

        */
    }

    #[test] fn q8gemm_8x8_aarch64_neon_k_gt_8_bzp0() {
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
                .bZeroPoint(0)
                .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
          }

        */
    }

    #[test] fn q8gemm_8x8_aarch64_neon_k_gt_8_nozp() {
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
                .aZeroPoint(0)
                .bZeroPoint(0)
                .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
          }

        */
    }

    #[test] fn q8gemm_8x8_aarch64_neon_k_gt_8_subtile() {
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
                    .iterations(3)
                    .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
              }
            }
          }

        */
    }

    #[test] fn q8gemm_8x8_aarch64_neon_k_div_8() {
        todo!();
        /*
        
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(k).test(
                pytorch_q8gemm_ukernel_8x8__aarch64_neon);
          }

        */
    }

    #[test] fn q8gemm_8x8_aarch64_neon_k_div_8_strided_a() {
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
                .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
          }

        */
    }

    #[test] fn q8gemm_8x8_aarch64_neon_k_div_8_strided_c() {
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
                .cStride(17)
                .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
          }

        */
    }

    #[test] fn q8gemm_8x8_aarch64_neon_k_div_8_subtile() {
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
                    .iterations(3)
                    .test(pytorch_q8gemm_ukernel_8x8__aarch64_neon);
              }
            }
          }

        */
    }

    //
    // Dynamic Quantization
    //

    #[test] fn q8gemm_dq_8x8_aarch64_neon_k_eq_8() {
        todo!();
        /*
        
          GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).test(
              pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);

        */
    }

    #[test] fn q8gemm_dq_8x8_aarch64_neon_k_eq_8_strided_a() {
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
              .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);

        */
    }

    #[test] fn q8gemm_dq_8x8_aarch64_neon_k_eq_8_strided_c() {
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
              .cStride(17)
              .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);

        */
    }

    #[test] fn q8gemm_dq_8x8_aarch64_neon_k_eq_8_qmin128() {
        todo!();
        /*
        
          GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).qmin(128).test(
              pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);

        */
    }

    #[test] fn q8gemm_dq_8x8_aarch64_neon_k_eq_8_qmax128() {
        todo!();
        /*
        
          GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).qmax(128).test(
              pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);

        */
    }

    #[test] fn q8gemm_dq_8x8_aarch64_neon_k_eq_8_azp0() {
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
              .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);

        */
    }

    #[test] fn q8gemm_dq_8x8_aarch64_neon_k_eq_8_bzp0() {
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
              .bZeroPoint(0)
              .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);

        */
    }

    #[test] fn q8gemm_dq_8x8_aarch64_neon_k_eq_8_nozp() {
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
              .bZeroPoint(0)
              .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);

        */
    }

    #[test] fn q8gemm_dq_8x8_aarch64_neon_k_gt_8() {
        todo!();
        /*
        
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(k).test(
                pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
          }

        */
    }

    #[test] fn q8gemm_dq_8x8_aarch64_neon_k_gt_8_strided_a() {
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
                .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
          }

        */
    }

    #[test] fn q8gemm_dq_8x8_aarch64_neon_k_gt_8_strided_c() {
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
                .cStride(17)
                .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
          }

        */
    }

    #[test] fn q8gemm_dq_8x8_aarch64_neon_k_gt_8_azp0() {
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
                .aZeroPoint(0)
                .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
          }

        */
    }

    #[test] fn q8gemm_dq_8x8_aarch64_neon_k_gt_8_bzp0() {
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
                .bZeroPoint(0)
                .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
          }

        */
    }

    #[test] fn q8gemm_dq_8x8_aarch64_neon_k_gt_8_nozp() {
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
                .aZeroPoint(0)
                .bZeroPoint(0)
                .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
          }

        */
    }

    #[test] fn q8gemm_dq_8x8_aarch64_neon_k_gt_8_subtile() {
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
                    .iterations(3)
                    .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
              }
            }
          }

        */
    }

    #[test] fn q8gemm_dq_8x8_aarch64_neon_k_div_8() {
        todo!();
        /*
        
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(k).test(
                pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
          }

        */
    }

    #[test] fn q8gemm_dq_8x8_aarch64_neon_k_div_8_strided_a() {
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
                .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
          }

        */
    }

    #[test] fn q8gemm_dq_8x8_aarch64_neon_k_div_8_strided_c() {
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
                .cStride(17)
                .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
          }

        */
    }

    #[test] fn q8gemm_dq_8x8_aarch64_neon_k_div_8_subtile() {
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
                    .iterations(3)
                    .test(pytorch_q8gemm_dq_ukernel_8x8__aarch64_neon);
              }
            }
          }

        */
    }
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod arm {

    use super::*;

    #[test] fn q8gemm_4x8_neon_k_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).test(
              pytorch_q8gemm_ukernel_4x8__neon);

        */
    }

    #[test] fn q8gemm_4x8_neon_k_eq_8_strided_a() {
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
              .test(pytorch_q8gemm_ukernel_4x8__neon);

        */
    }

    #[test] fn q8gemm_4x8_neon_k_eq_8_strided_c() {
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
              .cStride(17)
              .test(pytorch_q8gemm_ukernel_4x8__neon);

        */
    }

    #[test] fn q8gemm_4x8_neon_k_eq_8_qmin128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmin(128).test(
              pytorch_q8gemm_ukernel_4x8__neon);

        */
    }

    #[test] fn q8gemm_4x8_neon_k_eq_8_qmax128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmax(128).test(
              pytorch_q8gemm_ukernel_4x8__neon);

        */
    }

    #[test] fn q8gemm_4x8_neon_k_eq_8_azp0() {
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
              .test(pytorch_q8gemm_ukernel_4x8__neon);

        */
    }

    #[test] fn q8gemm_4x8_neon_k_eq_8_bzp0() {
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
              .bZeroPoint(0)
              .test(pytorch_q8gemm_ukernel_4x8__neon);

        */
    }

    #[test] fn q8gemm_4x8_neon_k_eq_8_nozp() {
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
              .bZeroPoint(0)
              .test(pytorch_q8gemm_ukernel_4x8__neon);

        */
    }

    #[test] fn q8gemm_4x8_neon_k_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(k).test(
                pytorch_q8gemm_ukernel_4x8__neon);
          }

        */
    }

    #[test] fn q8gemm_4x8_neon_k_gt_8_strided_a() {
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
                .test(pytorch_q8gemm_ukernel_4x8__neon);
          }

        */
    }

    #[test] fn q8gemm_4x8_neon_k_gt_8_strided_c() {
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
                .cStride(17)
                .test(pytorch_q8gemm_ukernel_4x8__neon);
          }

        */
    }

    #[test] fn q8gemm_4x8_neon_k_gt_8_azp0() {
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
                .aZeroPoint(0)
                .test(pytorch_q8gemm_ukernel_4x8__neon);
          }

        */
    }

    #[test] fn q8gemm_4x8_neon_k_gt_8_bzp0() {
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
                .bZeroPoint(0)
                .test(pytorch_q8gemm_ukernel_4x8__neon);
          }

        */
    }

    #[test] fn q8gemm_4x8_neon_k_gt_8_nozp() {
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
                .aZeroPoint(0)
                .bZeroPoint(0)
                .test(pytorch_q8gemm_ukernel_4x8__neon);
          }

        */
    }

    #[test] fn q8gemm_4x8_neon_k_gt_8_subtile() {
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
                    .iterations(3)
                    .test(pytorch_q8gemm_ukernel_4x8__neon);
              }
            }
          }

        */
    }

    #[test] fn q8gemm_4x8_neon_k_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(k).test(
                pytorch_q8gemm_ukernel_4x8__neon);
          }

        */
    }

    #[test] fn q8gemm_4x8_neon_k_div_8_strided_a() {
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
                .test(pytorch_q8gemm_ukernel_4x8__neon);
          }

        */
    }

    #[test] fn q8gemm_4x8_neon_k_div_8_strided_c() {
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
                .cStride(17)
                .test(pytorch_q8gemm_ukernel_4x8__neon);
          }

        */
    }

    #[test] fn q8gemm_4x8_neon_k_div_8_subtile() {
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
                    .iterations(3)
                    .test(pytorch_q8gemm_ukernel_4x8__neon);
              }
            }
          }

        */
    }

    //
    // Dynamic Quantization
    //

    #[test] fn q8gemm_dq_4x8_neon_k_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).test(
              pytorch_q8gemm_dq_ukernel_4x8__neon);

        */
    }

    #[test] fn q8gemm_dq_4x8_neon_k_eq_8_strided_a() {
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
              .test(pytorch_q8gemm_dq_ukernel_4x8__neon);

        */
    }

    #[test] fn q8gemm_dq_4x8_neon_k_eq_8_strided_c() {
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
              .cStride(17)
              .test(pytorch_q8gemm_dq_ukernel_4x8__neon);

        */
    }

    #[test] fn q8gemm_dq_4x8_neon_k_eq_8_qmin128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmin(128).test(
              pytorch_q8gemm_dq_ukernel_4x8__neon);

        */
    }

    #[test] fn q8gemm_dq_4x8_neon_k_eq_8_qmax128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmax(128).test(
              pytorch_q8gemm_dq_ukernel_4x8__neon);

        */
    }

    #[test] fn q8gemm_dq_4x8_neon_k_eq_8_azp0() {
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
              .test(pytorch_q8gemm_dq_ukernel_4x8__neon);

        */
    }

    #[test] fn q8gemm_dq_4x8_neon_k_eq_8_bzp0() {
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
              .bZeroPoint(0)
              .test(pytorch_q8gemm_dq_ukernel_4x8__neon);

        */
    }

    #[test] fn q8gemm_dq_4x8_neon_k_eq_8_nozp() {
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
              .bZeroPoint(0)
              .test(pytorch_q8gemm_dq_ukernel_4x8__neon);

        */
    }

    #[test] fn q8gemm_dq_4x8_neon_k_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(k).test(
                pytorch_q8gemm_dq_ukernel_4x8__neon);
          }

        */
    }

    #[test] fn q8gemm_dq_4x8_neon_k_gt_8_strided_a() {
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
                .test(pytorch_q8gemm_dq_ukernel_4x8__neon);
          }

        */
    }

    #[test] fn q8gemm_dq_4x8_neon_k_gt_8_strided_c() {
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
                .cStride(17)
                .test(pytorch_q8gemm_dq_ukernel_4x8__neon);
          }

        */
    }

    #[test] fn q8gemm_dq_4x8_neon_k_gt_8_azp0() {
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
                .aZeroPoint(0)
                .test(pytorch_q8gemm_dq_ukernel_4x8__neon);
          }

        */
    }

    #[test] fn q8gemm_dq_4x8_neon_k_gt_8_bzp0() {
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
                .bZeroPoint(0)
                .test(pytorch_q8gemm_dq_ukernel_4x8__neon);
          }

        */
    }

    #[test] fn q8gemm_dq_4x8_neon_k_gt_8_nozp() {
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
                .aZeroPoint(0)
                .bZeroPoint(0)
                .test(pytorch_q8gemm_dq_ukernel_4x8__neon);
          }

        */
    }

    #[test] fn q8gemm_dq_4x8_neon_k_gt_8_subtile() {
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
                    .iterations(3)
                    .test(pytorch_q8gemm_dq_ukernel_4x8__neon);
              }
            }
          }

        */
    }

    #[test] fn q8gemm_dq_4x8_neon_k_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(k).test(
                pytorch_q8gemm_dq_ukernel_4x8__neon);
          }

        */
    }

    #[test] fn q8gemm_dq_4x8_neon_k_div_8_strided_a() {
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
                .test(pytorch_q8gemm_dq_ukernel_4x8__neon);
          }

        */
    }

    #[test] fn q8gemm_dq_4x8_neon_k_div_8_strided_c() {
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
                .cStride(17)
                .test(pytorch_q8gemm_dq_ukernel_4x8__neon);
          }

        */
    }

    #[test] fn q8gemm_dq_4x8_neon_k_div_8_subtile() {
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
                    .iterations(3)
                    .test(pytorch_q8gemm_dq_ukernel_4x8__neon);
              }
            }
          }

        */
    }

    #[test] fn q8gemm_8x8_neon_k_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).test(
              pytorch_q8gemm_ukernel_8x8__neon);

        */
    }

    #[test] fn q8gemm_8x8_neon_k_eq_8_strided_a() {
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
              .test(pytorch_q8gemm_ukernel_8x8__neon);

        */
    }

    #[test] fn q8gemm_8x8_neon_k_eq_8_strided_c() {
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
              .cStride(17)
              .test(pytorch_q8gemm_ukernel_8x8__neon);

        */
    }

    #[test] fn q8gemm_8x8_neon_k_eq_8_qmin128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).qmin(128).test(
              pytorch_q8gemm_ukernel_8x8__neon);

        */
    }

    #[test] fn q8gemm_8x8_neon_k_eq_8_qmax128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(8).qmax(128).test(
              pytorch_q8gemm_ukernel_8x8__neon);

        */
    }

    #[test] fn q8gemm_8x8_neon_k_eq_8_azp0() {
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
              .test(pytorch_q8gemm_ukernel_8x8__neon);

        */
    }

    #[test] fn q8gemm_8x8_neon_k_eq_8_bzp0() {
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
              .bZeroPoint(0)
              .test(pytorch_q8gemm_ukernel_8x8__neon);

        */
    }

    #[test] fn q8gemm_8x8_neon_k_eq_8_nozp() {
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
              .bZeroPoint(0)
              .test(pytorch_q8gemm_ukernel_8x8__neon);

        */
    }

    #[test] fn q8gemm_8x8_neon_k_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(k).test(
                pytorch_q8gemm_ukernel_8x8__neon);
          }

        */
    }

    #[test] fn q8gemm_8x8_neon_k_gt_8_strided_a() {
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
                .test(pytorch_q8gemm_ukernel_8x8__neon);
          }

        */
    }

    #[test] fn q8gemm_8x8_neon_k_gt_8_strided_c() {
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
                .cStride(17)
                .test(pytorch_q8gemm_ukernel_8x8__neon);
          }

        */
    }

    #[test] fn q8gemm_8x8_neon_k_gt_8_azp0() {
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
                .aZeroPoint(0)
                .test(pytorch_q8gemm_ukernel_8x8__neon);
          }

        */
    }

    #[test] fn q8gemm_8x8_neon_k_gt_8_bzp0() {
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
                .bZeroPoint(0)
                .test(pytorch_q8gemm_ukernel_8x8__neon);
          }

        */
    }

    #[test] fn q8gemm_8x8_neon_k_gt_8_nozp() {
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
                .aZeroPoint(0)
                .bZeroPoint(0)
                .test(pytorch_q8gemm_ukernel_8x8__neon);
          }

        */
    }

    #[test] fn q8gemm_8x8_neon_k_gt_8_subtile() {
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
                    .iterations(3)
                    .test(pytorch_q8gemm_ukernel_8x8__neon);
              }
            }
          }

        */
    }

    #[test] fn q8gemm_8x8_neon_k_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(k).test(
                pytorch_q8gemm_ukernel_8x8__neon);
          }

        */
    }

    #[test] fn q8gemm_8x8_neon_k_div_8_strided_a() {
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
                .test(pytorch_q8gemm_ukernel_8x8__neon);
          }

        */
    }

    #[test] fn q8gemm_8x8_neon_k_div_8_strided_c() {
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
                .cStride(17)
                .test(pytorch_q8gemm_ukernel_8x8__neon);
          }

        */
    }

    #[test] fn q8gemm_8x8_neon_k_div_8_subtile() {
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
                    .iterations(3)
                    .test(pytorch_q8gemm_ukernel_8x8__neon);
              }
            }
          }

        */
    }

    #[test] fn q8gemm_6x4_neon_k_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(6).nr(4).np(4).kr(1).m(6).n(4).k(8).test(
              pytorch_q8gemm_ukernel_6x4__neon);

        */
    }

    #[test] fn q8gemm_6x4_neon_k_eq_8_strided_a() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(6)
              .nr(4)
              .np(4)
              .kr(1)
              .m(6)
              .n(4)
              .k(8)
              .aStride(37)
              .test(pytorch_q8gemm_ukernel_6x4__neon);

        */
    }

    #[test] fn q8gemm_6x4_neon_k_eq_8_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(6)
              .nr(4)
              .np(4)
              .kr(1)
              .m(6)
              .n(4)
              .k(8)
              .cStride(17)
              .test(pytorch_q8gemm_ukernel_6x4__neon);

        */
    }

    #[test] fn q8gemm_6x4_neon_k_eq_8_qmin128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(6).nr(4).np(4).kr(1).m(6).n(4).k(8).qmin(128).test(
              pytorch_q8gemm_ukernel_6x4__neon);

        */
    }

    #[test] fn q8gemm_6x4_neon_k_eq_8_qmax128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(6).nr(4).np(4).kr(1).m(6).n(4).k(8).qmax(128).test(
              pytorch_q8gemm_ukernel_6x4__neon);

        */
    }

    #[test] fn q8gemm_6x4_neon_k_eq_8_azp0() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(6)
              .nr(4)
              .np(4)
              .kr(1)
              .m(6)
              .n(4)
              .k(8)
              .aZeroPoint(0)
              .test(pytorch_q8gemm_ukernel_6x4__neon);

        */
    }

    #[test] fn q8gemm_6x4_neon_k_eq_8_bzp0() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(6)
              .nr(4)
              .np(4)
              .kr(1)
              .m(6)
              .n(4)
              .k(8)
              .bZeroPoint(0)
              .test(pytorch_q8gemm_ukernel_6x4__neon);

        */
    }

    #[test] fn q8gemm_6x4_neon_k_eq_8_nozp() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(6)
              .nr(4)
              .np(4)
              .kr(1)
              .m(6)
              .n(4)
              .k(8)
              .aZeroPoint(0)
              .bZeroPoint(0)
              .test(pytorch_q8gemm_ukernel_6x4__neon);

        */
    }

    #[test] fn q8gemm_6x4_neon_k_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester().mr(6).nr(4).np(4).kr(1).m(6).n(4).k(k).test(
                pytorch_q8gemm_ukernel_6x4__neon);
          }

        */
    }

    #[test] fn q8gemm_6x4_neon_k_gt_8_strided_a() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(6)
                .nr(4)
                .np(4)
                .kr(1)
                .m(6)
                .n(4)
                .k(k)
                .aStride(37)
                .test(pytorch_q8gemm_ukernel_6x4__neon);
          }

        */
    }

    #[test] fn q8gemm_6x4_neon_k_gt_8_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(6)
                .nr(4)
                .np(4)
                .kr(1)
                .m(6)
                .n(4)
                .k(k)
                .cStride(17)
                .test(pytorch_q8gemm_ukernel_6x4__neon);
          }

        */
    }

    #[test] fn q8gemm_6x4_neon_k_gt_8_azp0() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(6)
                .nr(4)
                .np(4)
                .kr(1)
                .m(6)
                .n(4)
                .k(k)
                .aZeroPoint(0)
                .test(pytorch_q8gemm_ukernel_6x4__neon);
          }

        */
    }

    #[test] fn q8gemm_6x4_neon_k_gt_8_bzp0() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(6)
                .nr(4)
                .np(4)
                .kr(1)
                .m(6)
                .n(4)
                .k(k)
                .bZeroPoint(0)
                .test(pytorch_q8gemm_ukernel_6x4__neon);
          }

        */
    }

    #[test] fn q8gemm_6x4_neon_k_gt_8_nozp() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(6)
                .nr(4)
                .np(4)
                .kr(1)
                .m(6)
                .n(4)
                .k(k)
                .aZeroPoint(0)
                .bZeroPoint(0)
                .test(pytorch_q8gemm_ukernel_6x4__neon);
          }

        */
    }

    #[test] fn q8gemm_6x4_neon_k_gt_8_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            for (u32 m = 1; m <= 6; m++) {
              for (u32 n = 1; n <= 4; n++) {
                GemmMicrokernelTester()
                    .mr(6)
                    .nr(4)
                    .np(4)
                    .kr(1)
                    .m(m)
                    .n(n)
                    .k(k)
                    .iterations(3)
                    .test(pytorch_q8gemm_ukernel_6x4__neon);
              }
            }
          }

        */
    }

    #[test] fn q8gemm_6x4_neon_k_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester().mr(6).nr(4).np(4).kr(1).m(6).n(4).k(k).test(
                pytorch_q8gemm_ukernel_6x4__neon);
          }

        */
    }

    #[test] fn q8gemm_6x4_neon_k_div_8_strided_a() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester()
                .mr(6)
                .nr(4)
                .np(4)
                .kr(1)
                .m(6)
                .n(4)
                .k(k)
                .aStride(171)
                .test(pytorch_q8gemm_ukernel_6x4__neon);
          }

        */
    }

    #[test] fn q8gemm_6x4_neon_k_div_8_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester()
                .mr(6)
                .nr(4)
                .np(4)
                .kr(1)
                .m(6)
                .n(4)
                .k(k)
                .cStride(17)
                .test(pytorch_q8gemm_ukernel_6x4__neon);
          }

        */
    }

    #[test] fn q8gemm_6x4_neon_k_div_8_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 16; k < 128; k += 24) {
            for (u32 m = 1; m <= 6; m++) {
              for (u32 n = 1; n <= 4; n++) {
                GemmMicrokernelTester().mr(6).nr(4).np(4).kr(1).m(m).n(n).k(k).test(
                    pytorch_q8gemm_ukernel_6x4__neon);
              }
            }
          }

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_neon_k_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(4).nr(8).np(8).kr(2).m(4).n(8).k(8).test(
              pytorch_q8gemm_xzp_ukernel_4x8c2__neon);

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_neon_k_eq_8_strided_a() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(4)
              .nr(8)
              .np(8)
              .kr(2)
              .m(4)
              .n(8)
              .k(8)
              .aStride(37)
              .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_neon_k_eq_8_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(4)
              .nr(8)
              .np(8)
              .kr(2)
              .m(4)
              .n(8)
              .k(8)
              .cStride(17)
              .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_neon_k_eq_8_qmin128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(4).nr(8).np(8).kr(2).m(4).n(8).k(8).qmin(128).test(
              pytorch_q8gemm_xzp_ukernel_4x8c2__neon);

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_neon_k_eq_8_qmax128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(4).nr(8).np(8).kr(2).m(4).n(8).k(8).qmax(128).test(
              pytorch_q8gemm_xzp_ukernel_4x8c2__neon);

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_neon_k_eq_8_azp0() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(4)
              .nr(8)
              .np(8)
              .kr(2)
              .m(4)
              .n(8)
              .k(8)
              .aZeroPoint(0)
              .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_neon_k_eq_8_bzp0() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(4)
              .nr(8)
              .np(8)
              .kr(2)
              .m(4)
              .n(8)
              .k(8)
              .bZeroPoint(0)
              .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_neon_k_eq_8_nozp() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(4)
              .nr(8)
              .np(8)
              .kr(2)
              .m(4)
              .n(8)
              .k(8)
              .aZeroPoint(0)
              .bZeroPoint(0)
              .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_neon_k_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester().mr(4).nr(8).np(8).kr(2).m(4).n(8).k(k).test(
                pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
          }

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_neon_k_gt_8_strided_a() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(8)
                .np(8)
                .kr(2)
                .m(4)
                .n(8)
                .k(k)
                .aStride(37)
                .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
          }

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_neon_k_gt_8_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(8)
                .np(8)
                .kr(2)
                .m(4)
                .n(8)
                .k(k)
                .cStride(17)
                .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
          }

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_neon_k_gt_8_azp0() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(8)
                .np(8)
                .kr(2)
                .m(4)
                .n(8)
                .k(k)
                .aZeroPoint(0)
                .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
          }

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_neon_k_gt_8_bzp0() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(8)
                .np(8)
                .kr(2)
                .m(4)
                .n(8)
                .k(k)
                .bZeroPoint(0)
                .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
          }

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_neon_k_gt_8_nozp() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(8)
                .np(8)
                .kr(2)
                .m(4)
                .n(8)
                .k(k)
                .aZeroPoint(0)
                .bZeroPoint(0)
                .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
          }

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_neon_k_gt_8_subtile() {
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
                    .kr(2)
                    .m(m)
                    .n(n)
                    .k(k)
                    .iterations(3)
                    .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
              }
            }
          }

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_neon_k_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester().mr(4).nr(8).np(8).kr(2).m(4).n(8).k(k).test(
                pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
          }

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_neon_k_div_8_strided_a() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(8)
                .np(8)
                .kr(2)
                .m(4)
                .n(8)
                .k(k)
                .aStride(171)
                .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
          }

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_neon_k_div_8_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester()
                .mr(4)
                .nr(8)
                .np(8)
                .kr(2)
                .m(4)
                .n(8)
                .k(k)
                .cStride(17)
                .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
          }

        */
    }

    #[test] fn q8gemm_4x8c2_xzp_neon_k_div_8_subtile() {
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
                    .kr(2)
                    .m(m)
                    .n(n)
                    .k(k)
                    .iterations(3)
                    .test(pytorch_q8gemm_xzp_ukernel_4x8c2__neon);
              }
            }
          }

        */
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;

    #[test] fn q8gemm_2x4c8_sse2_k_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester().mr(2).nr(4).np(1).kr(8).m(2).n(4).k(8).test(
              pytorch_q8gemm_ukernel_2x4c8__sse2);

        */
    }

    #[test] fn q8gemm_2x4c8_sse2_k_eq_8_strided_a() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester()
              .mr(2)
              .nr(4)
              .np(1)
              .kr(8)
              .m(2)
              .n(4)
              .k(8)
              .aStride(37)
              .test(pytorch_q8gemm_ukernel_2x4c8__sse2);

        */
    }

    #[test] fn q8gemm_2x4c8_sse2_k_eq_8_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester()
              .mr(2)
              .nr(4)
              .np(1)
              .kr(8)
              .m(2)
              .n(4)
              .k(8)
              .cStride(17)
              .test(pytorch_q8gemm_ukernel_2x4c8__sse2);

        */
    }

    #[test] fn q8gemm_2x4c8_sse2_k_eq_8_qmin128() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester().mr(2).nr(4).np(1).kr(8).m(2).n(4).k(8).qmin(128).test(
              pytorch_q8gemm_ukernel_2x4c8__sse2);

        */
    }

    #[test] fn q8gemm_2x4c8_sse2_k_eq_8_qmax128() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester().mr(2).nr(4).np(1).kr(8).m(2).n(4).k(8).qmax(128).test(
              pytorch_q8gemm_ukernel_2x4c8__sse2);

        */
    }

    #[test] fn q8gemm_2x4c8_sse2_k_eq_8_azp0() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester()
              .mr(2)
              .nr(4)
              .np(1)
              .kr(8)
              .m(2)
              .n(4)
              .k(8)
              .aZeroPoint(0)
              .test(pytorch_q8gemm_ukernel_2x4c8__sse2);

        */
    }

    #[test] fn q8gemm_2x4c8_sse2_k_eq_8_bzp0() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester()
              .mr(2)
              .nr(4)
              .np(1)
              .kr(8)
              .m(2)
              .n(4)
              .k(8)
              .bZeroPoint(0)
              .test(pytorch_q8gemm_ukernel_2x4c8__sse2);

        */
    }

    #[test] fn q8gemm_2x4c8_sse2_k_eq_8_nozp() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester()
              .mr(2)
              .nr(4)
              .np(1)
              .kr(8)
              .m(2)
              .n(4)
              .k(8)
              .aZeroPoint(0)
              .bZeroPoint(0)
              .test(pytorch_q8gemm_ukernel_2x4c8__sse2);

        */
    }

    #[test] fn q8gemm_2x4c8_sse2_k_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester().mr(2).nr(4).np(1).kr(8).m(2).n(4).k(k).test(
                pytorch_q8gemm_ukernel_2x4c8__sse2);
          }

        */
    }

    #[test] fn q8gemm_2x4c8_sse2_k_gt_8_strided_a() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(2)
                .nr(4)
                .np(1)
                .kr(8)
                .m(2)
                .n(4)
                .k(k)
                .aStride(37)
                .test(pytorch_q8gemm_ukernel_2x4c8__sse2);
          }

        */
    }

    #[test] fn q8gemm_2x4c8_sse2_k_gt_8_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(2)
                .nr(4)
                .np(1)
                .kr(8)
                .m(2)
                .n(4)
                .k(k)
                .cStride(17)
                .test(pytorch_q8gemm_ukernel_2x4c8__sse2);
          }

        */
    }

    #[test] fn q8gemm_2x4c8_sse2_k_gt_8_azp0() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(2)
                .nr(4)
                .np(1)
                .kr(8)
                .m(2)
                .n(4)
                .k(k)
                .aZeroPoint(0)
                .test(pytorch_q8gemm_ukernel_2x4c8__sse2);
          }

        */
    }

    #[test] fn q8gemm_2x4c8_sse2_k_gt_8_bzp0() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(2)
                .nr(4)
                .np(1)
                .kr(8)
                .m(2)
                .n(4)
                .k(k)
                .bZeroPoint(0)
                .test(pytorch_q8gemm_ukernel_2x4c8__sse2);
          }

        */
    }

    #[test] fn q8gemm_2x4c8_sse2_k_gt_8_nozp() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(2)
                .nr(4)
                .np(1)
                .kr(8)
                .m(2)
                .n(4)
                .k(k)
                .aZeroPoint(0)
                .bZeroPoint(0)
                .test(pytorch_q8gemm_ukernel_2x4c8__sse2);
          }

        */
    }

    #[test] fn q8gemm_2x4c8_sse2_k_gt_8_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize k = 9; k < 16; k++) {
            for (u32 m = 1; m <= 2; m++) {
              for (u32 n = 1; n <= 4; n++) {
                GemmMicrokernelTester()
                    .mr(2)
                    .nr(4)
                    .np(1)
                    .kr(8)
                    .m(m)
                    .n(n)
                    .k(k)
                    .iterations(3)
                    .test(pytorch_q8gemm_ukernel_2x4c8__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8gemm_2x4c8_sse2_k_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester().mr(2).nr(4).np(1).kr(8).m(2).n(4).k(k).test(
                pytorch_q8gemm_ukernel_2x4c8__sse2);
          }

        */
    }

    #[test] fn q8gemm_2x4c8_sse2_k_div_8_strided_a() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester()
                .mr(2)
                .nr(4)
                .np(1)
                .kr(8)
                .m(2)
                .n(4)
                .k(k)
                .aStride(171)
                .test(pytorch_q8gemm_ukernel_2x4c8__sse2);
          }

        */
    }

    #[test] fn q8gemm_2x4c8_sse2_k_div_8_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester()
                .mr(2)
                .nr(4)
                .np(1)
                .kr(8)
                .m(2)
                .n(4)
                .k(k)
                .cStride(17)
                .test(pytorch_q8gemm_ukernel_2x4c8__sse2);
          }

        */
    }

    #[test] fn q8gemm_2x4c8_sse2_k_div_8_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize k = 16; k < 128; k += 24) {
            for (u32 m = 1; m <= 2; m++) {
              for (u32 n = 1; n <= 4; n++) {
                GemmMicrokernelTester()
                    .mr(2)
                    .nr(4)
                    .np(1)
                    .kr(8)
                    .m(m)
                    .n(n)
                    .k(k)
                    .iterations(3)
                    .test(pytorch_q8gemm_ukernel_2x4c8__sse2);
              }
            }
          }

        */
    }

    /**
      | Following tests fail both on original
      | QNNPack and the version with runtime
      | requantization.
      |
      */
    #[cfg(feature = "run-failing-tests")]
    pub mod failing_tests {

        use super::*;

        #[test] fn q8gemm_4x4c2_sse2_k_eq_1() {
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
                      .k(1)
                      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
                  
            */
        }

        #[test] fn q8gemm_4x4c2_sse2_k_eq_1_strided_a() {
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
                      .k(1)
                      .aStride(37)
                      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
                  
            */
        }

        #[test] fn q8gemm_4x4c2_sse2_k_eq_1_strided_c() {
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
                      .k(1)
                      .cStride(17)
                      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
                  
            */
        }

        #[test] fn q8gemm_4x4c2_sse2_k_eq_1_qmin128() {
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
                      .k(1)
                      .qmin(128)
                      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
                  
            */
        }

        #[test] fn q8gemm_4x4c2_sse2_k_eq_1_qmax128() {

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
                      .k(1)
                      .qmax(128)
                      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
                  
            */
        }

        #[test] fn q8gemm_4x4c2_sse2_k_eq_1_azp0() {
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
                      .k(1)
                      .aZeroPoint(0)
                      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
                  
            */
        }

        #[test] fn q8gemm_4x4c2_sse2_k_eq_1_bzp0() {
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
                      .k(1)
                      .bZeroPoint(0)
                      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
                  
            */
        }

        #[test] fn q8gemm_4x4c2_sse2_k_eq_1_nozp() {
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
                      .k(1)
                      .aZeroPoint(0)
                      .bZeroPoint(0)
                      .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
                  
            */
        }
    }

    #[test] fn q8gemm_4x4c2_sse2_k_lt_4() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(3).test(
              pytorch_q8gemm_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_lt_4_strided_a() {
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
              .k(3)
              .aStride(37)
              .test(pytorch_q8gemm_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_lt_4_strided_c() {
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
              .k(3)
              .cStride(17)
              .test(pytorch_q8gemm_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_lt_4_qmin128() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(3).qmin(128).test(
              pytorch_q8gemm_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_lt_4_qmax128() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(3).qmax(128).test(
              pytorch_q8gemm_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_lt_4_azp0() {
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
              .k(3)
              .aZeroPoint(0)
              .test(pytorch_q8gemm_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_lt_4_bzp0() {
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
              .k(3)
              .bZeroPoint(0)
              .test(pytorch_q8gemm_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_lt_4_nozp() {
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
              .k(3)
              .aZeroPoint(0)
              .bZeroPoint(0)
              .test(pytorch_q8gemm_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_lt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(5).test(
              pytorch_q8gemm_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_lt_8_strided_a() {
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
              .k(5)
              .aStride(37)
              .test(pytorch_q8gemm_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_lt_8_strided_c() {
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
              .k(5)
              .cStride(17)
              .test(pytorch_q8gemm_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_lt_8_qmin128() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(5).qmin(128).test(
              pytorch_q8gemm_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_lt_8_qmax128() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(5).qmax(128).test(
              pytorch_q8gemm_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_lt_8_azp0() {
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
              .k(5)
              .aZeroPoint(0)
              .test(pytorch_q8gemm_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_lt_8_bzp0() {
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
              .k(5)
              .bZeroPoint(0)
              .test(pytorch_q8gemm_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_lt_8_nozp() {
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
              .k(5)
              .aZeroPoint(0)
              .bZeroPoint(0)
              .test(pytorch_q8gemm_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(8).test(
              pytorch_q8gemm_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_eq_8_strided_a() {
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
              .test(pytorch_q8gemm_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_eq_8_strided_c() {
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
              .cStride(17)
              .test(pytorch_q8gemm_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_eq_8_qmin128() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(8).qmin(128).test(
              pytorch_q8gemm_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_eq_8_qmax128() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(8).qmax(128).test(
              pytorch_q8gemm_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_eq_8_azp0() {
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
              .test(pytorch_q8gemm_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_eq_8_bzp0() {
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
              .bZeroPoint(0)
              .test(pytorch_q8gemm_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_eq_8_nozp() {
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
              .bZeroPoint(0)
              .test(pytorch_q8gemm_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(k).test(
                pytorch_q8gemm_ukernel_4x4c2__sse2);
          }

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_gt_8_strided_a() {
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
                .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
          }

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_gt_8_strided_c() {
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
                .cStride(17)
                .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
          }

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_gt_8_azp0() {
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
                .aZeroPoint(0)
                .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
          }

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_gt_8_bzp0() {
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
                .bZeroPoint(0)
                .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
          }

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_gt_8_nozp() {
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
                .aZeroPoint(0)
                .bZeroPoint(0)
                .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
          }

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_gt_8_subtile() {
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
                    .iterations(3)
                    .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(k).test(
                pytorch_q8gemm_ukernel_4x4c2__sse2);
          }

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_div_8_strided_a() {
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
                .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
          }

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_div_8_strided_c() {
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
                .cStride(17)
                .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
          }

        */
    }

    #[test] fn q8gemm_4x4c2_sse2_k_div_8_subtile() {
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
                    .iterations(3)
                    .test(pytorch_q8gemm_ukernel_4x4c2__sse2);
              }
            }
          }

        */
    }

    //
    // Dynamic Quantization
    //

    #[test] fn q8gemm_dq_4x4c2_sse2_k_lt_4() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(3).test(
              pytorch_q8gemm_dq_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_lt_4_strided_a() {
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
              .k(3)
              .aStride(37)
              .test(pytorch_q8gemm_dq_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_lt_4_strided_c() {
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
              .k(3)
              .cStride(17)
              .test(pytorch_q8gemm_dq_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_lt_4_qmin128() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(3).qmin(128).test(
              pytorch_q8gemm_dq_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_lt_4_qmax128() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(3).qmax(128).test(
              pytorch_q8gemm_dq_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_lt_4_azp0() {
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
              .k(3)
              .aZeroPoint(0)
              .test(pytorch_q8gemm_dq_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_lt_4_bzp0() {
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
              .k(3)
              .bZeroPoint(0)
              .test(pytorch_q8gemm_dq_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_lt_4_nozp() {
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
              .k(3)
              .aZeroPoint(0)
              .bZeroPoint(0)
              .test(pytorch_q8gemm_dq_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_lt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(5).test(
              pytorch_q8gemm_dq_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_lt_8_strided_a() {
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
              .k(5)
              .aStride(37)
              .test(pytorch_q8gemm_dq_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_lt_8_strided_c() {
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
              .k(5)
              .cStride(17)
              .test(pytorch_q8gemm_dq_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_lt_8_qmin128() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(5).qmin(128).test(
              pytorch_q8gemm_dq_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_lt_8_qmax128() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(5).qmax(128).test(
              pytorch_q8gemm_dq_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_lt_8_azp0() {
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
              .k(5)
              .aZeroPoint(0)
              .test(pytorch_q8gemm_dq_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_lt_8_bzp0() {
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
              .k(5)
              .bZeroPoint(0)
              .test(pytorch_q8gemm_dq_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_lt_8_nozp() {
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
              .k(5)
              .aZeroPoint(0)
              .bZeroPoint(0)
              .test(pytorch_q8gemm_dq_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_eq_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(8).test(
              pytorch_q8gemm_dq_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_eq_8_strided_a() {
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
              .test(pytorch_q8gemm_dq_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_eq_8_strided_c() {
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
              .cStride(17)
              .test(pytorch_q8gemm_dq_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_eq_8_qmin128() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(8).qmin(128).test(
              pytorch_q8gemm_dq_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_eq_8_qmax128() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(8).qmax(128).test(
              pytorch_q8gemm_dq_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_eq_8_azp0() {
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
              .test(pytorch_q8gemm_dq_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_eq_8_bzp0() {
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
              .bZeroPoint(0)
              .test(pytorch_q8gemm_dq_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_eq_8_nozp() {
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
              .bZeroPoint(0)
              .test(pytorch_q8gemm_dq_ukernel_4x4c2__sse2);

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_gt_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize k = 9; k < 16; k++) {
            GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(k).test(
                pytorch_q8gemm_dq_ukernel_4x4c2__sse2);
          }

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_gt_8_strided_a() {
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
                .test(pytorch_q8gemm_dq_ukernel_4x4c2__sse2);
          }

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_gt_8_strided_c() {
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
                .cStride(17)
                .test(pytorch_q8gemm_dq_ukernel_4x4c2__sse2);
          }

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_gt_8_azp0() {
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
                .aZeroPoint(0)
                .test(pytorch_q8gemm_dq_ukernel_4x4c2__sse2);
          }

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_gt_8_bzp0() {
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
                .bZeroPoint(0)
                .test(pytorch_q8gemm_dq_ukernel_4x4c2__sse2);
          }

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_gt_8_nozp() {
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
                .aZeroPoint(0)
                .bZeroPoint(0)
                .test(pytorch_q8gemm_dq_ukernel_4x4c2__sse2);
          }

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_gt_8_subtile() {
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
                    .iterations(3)
                    .test(pytorch_q8gemm_dq_ukernel_4x4c2__sse2);
              }
            }
          }

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_div_8() {
        todo!();
        /*
        
          TEST_REQUIRES_X86_SSE2;
          for (usize k = 16; k < 128; k += 8) {
            GemmMicrokernelTester().mr(4).nr(4).np(4).kr(2).m(4).n(4).k(k).test(
                pytorch_q8gemm_dq_ukernel_4x4c2__sse2);
          }

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_div_8_strided_a() {
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
                .test(pytorch_q8gemm_dq_ukernel_4x4c2__sse2);
          }

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_div_8_strided_c() {
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
                .cStride(17)
                .test(pytorch_q8gemm_dq_ukernel_4x4c2__sse2);
          }

        */
    }

    #[test] fn q8gemm_dq_4x4c2_sse2_k_div_8_subtile() {
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
                    .iterations(3)
                    .test(pytorch_q8gemm_dq_ukernel_4x4c2__sse2);
              }
            }
          }

        */
    }
}
