// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/hgemm.cc]

#[cfg(CPUINFO_ARCH_ARM)]
mod cpuinfo_arch_arm {
    use super::*;

    #[test] fn hgemm_8x8_aarch32_neonfp16arith_k_eq_4() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(4).test(
              pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);

        */
    }

    #[test] fn hgemm_8x8_aarch32_neonfp16arith_k_eq_4_strided_a() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          GemmMicrokernelTester()
              .mr(8)
              .nr(8)
              .np(8)
              .kr(1)
              .m(8)
              .n(8)
              .k(4)
              .aStride(37)
              .test(pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);

        */
    }

    #[test] fn hgemm_8x8_aarch32_neonfp16arith_k_eq_4_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          GemmMicrokernelTester()
              .mr(8)
              .nr(8)
              .np(8)
              .kr(1)
              .m(8)
              .n(8)
              .k(4)
              .cStride(17)
              .test(pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);

        */
    }

    #[test] fn hgemm_8x8_aarch32_neonfp16arith_k_eq_4_qmin128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(4).qmin(128).test(
              pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);

        */
    }

    #[test] fn hgemm_8x8_aarch32_neonfp16arith_k_eq_4_qmax128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(4).qmax(128).test(
              pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);

        */
    }

    #[test] fn hgemm_8x8_aarch32_neonfp16arith_k_gt_4() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          for (usize k = 5; k < 8; k++) {
            GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(k).test(
                pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
          }

        */
    }

    #[test] fn hgemm_8x8_aarch32_neonfp16arith_k_gt_4_strided_a() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          for (usize k = 5; k < 8; k++) {
            GemmMicrokernelTester()
                .mr(8)
                .nr(8)
                .np(8)
                .kr(1)
                .m(8)
                .n(8)
                .k(k)
                .aStride(37)
                .test(pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
          }

        */
    }

    #[test] fn hgemm_8x8_aarch32_neonfp16arith_k_gt_4_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          for (usize k = 5; k < 8; k++) {
            GemmMicrokernelTester()
                .mr(8)
                .nr(8)
                .np(8)
                .kr(1)
                .m(8)
                .n(8)
                .k(k)
                .cStride(17)
                .test(pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
          }

        */
    }

    #[test] fn hgemm_8x8_aarch32_neonfp16arith_k_gt_4_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          for (usize k = 5; k < 8; k++) {
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
                    .test(pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
              }
            }
          }

        */
    }

    #[test] fn hgemm_8x8_aarch32_neonfp16arith_k_div_4() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          for (usize k = 8; k < 64; k += 4) {
            GemmMicrokernelTester().mr(8).nr(8).np(8).kr(1).m(8).n(8).k(k).test(
                pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
          }

        */
    }

    #[test] fn hgemm_8x8_aarch32_neonfp16arith_k_div_4_strided_a() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          for (usize k = 8; k < 64; k += 4) {
            GemmMicrokernelTester()
                .mr(8)
                .nr(8)
                .np(8)
                .kr(1)
                .m(8)
                .n(8)
                .k(k)
                .aStride(171)
                .test(pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
          }

        */
    }

    #[test] fn hgemm_8x8_aarch32_neonfp16arith_k_div_4_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          for (usize k = 8; k < 64; k += 4) {
            GemmMicrokernelTester()
                .mr(8)
                .nr(8)
                .np(8)
                .kr(1)
                .m(8)
                .n(8)
                .k(k)
                .cStride(17)
                .test(pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
          }

        */
    }

    #[test] fn hgemm_8x8_aarch32_neonfp16arith_k_div_4_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          for (usize k = 8; k < 64; k += 12) {
            for (u32 m = 1; m <= 1; m++) {
              for (u32 n = 8; n <= 8; n++) {
                GemmMicrokernelTester()
                    .mr(8)
                    .nr(8)
                    .np(8)
                    .kr(1)
                    .m(m)
                    .n(n)
                    .k(k)
                    .iterations(3)
                    .test(pytorch_hgemm_ukernel_8x8__aarch32_neonfp16arith);
              }
            }
          }

        */
    }

}
