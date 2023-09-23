crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/sgemm.cc]

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod arch_arm {

    use super::*;

    #[test] fn sgemm_5x8_neon_k_eq_2() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(5).nr(8).np(8).kr(1).m(5).n(8).k(2).test(
              pytorch_sgemm_ukernel_5x8__neon);

        */
    }

    #[test] fn sgemm_5x8_neon_k_eq_2_strided_a() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(5)
              .nr(8)
              .np(8)
              .kr(1)
              .m(5)
              .n(8)
              .k(2)
              .aStride(37)
              .test(pytorch_sgemm_ukernel_5x8__neon);

        */
    }

    #[test] fn sgemm_5x8_neon_k_eq_2_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(5)
              .nr(8)
              .np(8)
              .kr(1)
              .m(5)
              .n(8)
              .k(2)
              .cStride(17)
              .test(pytorch_sgemm_ukernel_5x8__neon);

        */
    }

    #[test] fn sgemm_5x8_neon_k_eq_8_rmin128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmin(128).test(
              pytorch_sgemm_ukernel_5x8__neon);

        */
    }

    #[test] fn sgemm_5x8_neon_k_eq_8_qmax128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(8).qmax(128).test(
              pytorch_sgemm_ukernel_5x8__neon);

        */
    }

    #[test] fn sgemm_5x8_neon_k_gt_2() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 3; k < 16; k++) {
            GemmMicrokernelTester().mr(5).nr(8).np(8).kr(1).m(5).n(8).k(k).test(
                pytorch_sgemm_ukernel_5x8__neon);
          }

        */
    }

    #[test] fn sgemm_5x8_neon_k_gt_2_strided_a() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 3; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(5)
                .nr(8)
                .np(8)
                .kr(1)
                .m(5)
                .n(8)
                .k(k)
                .aStride(37)
                .test(pytorch_sgemm_ukernel_5x8__neon);
          }

        */
    }

    #[test] fn sgemm_5x8_neon_k_gt_2_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 3; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(5)
                .nr(8)
                .np(8)
                .kr(1)
                .m(5)
                .n(8)
                .k(k)
                .cStride(17)
                .test(pytorch_sgemm_ukernel_5x8__neon);
          }

        */
    }

    #[test] fn sgemm_5x8_neon_k_gt_2_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 3; k < 16; k++) {
            for (u32 m = 1; m <= 5; m++) {
              for (u32 n = 1; n <= 8; n++) {
                GemmMicrokernelTester()
                    .mr(5)
                    .nr(8)
                    .np(8)
                    .kr(1)
                    .m(m)
                    .n(n)
                    .k(k)
                    .iterations(3)
                    .test(pytorch_sgemm_ukernel_5x8__neon);
              }
            }
          }

        */
    }

    #[test] fn sgemm_5x8_neon_k_div_2() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 2; k < 32; k += 2) {
            GemmMicrokernelTester().mr(5).nr(8).np(8).kr(1).m(5).n(8).k(k).test(
                pytorch_sgemm_ukernel_5x8__neon);
          }

        */
    }

    #[test] fn sgemm_5x8_neon_k_div_2_strided_a() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 2; k < 32; k += 2) {
            GemmMicrokernelTester()
                .mr(5)
                .nr(8)
                .np(8)
                .kr(1)
                .m(5)
                .n(8)
                .k(k)
                .aStride(171)
                .test(pytorch_sgemm_ukernel_5x8__neon);
          }

        */
    }

    #[test] fn sgemm_5x8_neon_k_div_2_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 2; k < 32; k += 2) {
            GemmMicrokernelTester()
                .mr(5)
                .nr(8)
                .np(8)
                .kr(1)
                .m(5)
                .n(8)
                .k(k)
                .cStride(17)
                .test(pytorch_sgemm_ukernel_5x8__neon);
          }

        */
    }

    #[test] fn sgemm_5x8_neon_k_div_2_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 2; k < 32; k += 6) {
            for (u32 m = 1; m <= 5; m++) {
              for (u32 n = 1; n <= 8; n++) {
                GemmMicrokernelTester()
                    .mr(5)
                    .nr(8)
                    .np(8)
                    .kr(1)
                    .m(m)
                    .n(n)
                    .k(k)
                    .iterations(3)
                    .test(pytorch_sgemm_ukernel_5x8__neon);
              }
            }
          }

        */
    }

    #[test] fn sgemm_6x8_neon_k_eq_2() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(6).nr(8).np(8).kr(1).m(6).n(8).k(2).test(
              pytorch_sgemm_ukernel_6x8__neon);

        */
    }

    #[test] fn sgemm_6x8_neon_k_eq_2_strided_a() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(6)
              .nr(8)
              .np(8)
              .kr(1)
              .m(6)
              .n(8)
              .k(2)
              .aStride(37)
              .test(pytorch_sgemm_ukernel_6x8__neon);

        */
    }

    #[test] fn sgemm_6x8_neon_k_eq_2_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester()
              .mr(6)
              .nr(8)
              .np(8)
              .kr(1)
              .m(6)
              .n(8)
              .k(2)
              .cStride(17)
              .test(pytorch_sgemm_ukernel_6x8__neon);

        */
    }

    #[test] fn sgemm_6x8_neon_k_eq_8_qmin128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(6).nr(8).np(8).kr(1).m(6).n(8).k(8).qmin(128).test(
              pytorch_sgemm_ukernel_6x8__neon);

        */
    }

    #[test] fn sgemm_6x8_neon_k_eq_8_qmax128() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          GemmMicrokernelTester().mr(6).nr(8).np(8).kr(1).m(6).n(8).k(8).qmax(128).test(
              pytorch_sgemm_ukernel_6x8__neon);

        */
    }

    #[test] fn sgemm_6x8_neon_k_gt_2() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 3; k < 16; k++) {
            GemmMicrokernelTester().mr(6).nr(8).np(8).kr(1).m(6).n(8).k(k).test(
                pytorch_sgemm_ukernel_6x8__neon);
          }

        */
    }

    #[test] fn sgemm_6x8_neon_k_gt_2_strided_a() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 3; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(6)
                .nr(8)
                .np(8)
                .kr(1)
                .m(6)
                .n(8)
                .k(k)
                .aStride(37)
                .test(pytorch_sgemm_ukernel_6x8__neon);
          }

        */
    }

    #[test] fn sgemm_6x8_neon_k_gt_2_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 3; k < 16; k++) {
            GemmMicrokernelTester()
                .mr(6)
                .nr(8)
                .np(8)
                .kr(1)
                .m(6)
                .n(8)
                .k(k)
                .cStride(17)
                .test(pytorch_sgemm_ukernel_6x8__neon);
          }

        */
    }

    #[test] fn sgemm_6x8_neon_k_gt_2_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 3; k < 16; k++) {
            for (u32 m = 1; m <= 6; m++) {
              for (u32 n = 1; n <= 8; n++) {
                GemmMicrokernelTester()
                    .mr(6)
                    .nr(8)
                    .np(8)
                    .kr(1)
                    .m(m)
                    .n(n)
                    .k(k)
                    .iterations(3)
                    .test(pytorch_sgemm_ukernel_6x8__neon);
              }
            }
          }

        */
    }

    #[test] fn sgemm_6x8_neon_k_div_2() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 2; k < 32; k += 2) {
            GemmMicrokernelTester().mr(6).nr(8).np(8).kr(1).m(6).n(8).k(k).test(
                pytorch_sgemm_ukernel_6x8__neon);
          }

        */
    }

    #[test] fn sgemm_6x8_neon_k_div_2_strided_a() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 2; k < 32; k += 2) {
            GemmMicrokernelTester()
                .mr(6)
                .nr(8)
                .np(8)
                .kr(1)
                .m(6)
                .n(8)
                .k(k)
                .aStride(171)
                .test(pytorch_sgemm_ukernel_6x8__neon);
          }

        */
    }

    #[test] fn sgemm_6x8_neon_k_div_2_strided_c() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 2; k < 32; k += 2) {
            GemmMicrokernelTester()
                .mr(6)
                .nr(8)
                .np(8)
                .kr(1)
                .m(6)
                .n(8)
                .k(k)
                .cStride(17)
                .test(pytorch_sgemm_ukernel_6x8__neon);
          }

        */
    }

    #[test] fn sgemm_6x8_neon_k_div_2_subtile() {
        todo!();
        /*
        
          TEST_REQUIRES_ARM_NEON;
          for (usize k = 2; k < 32; k += 6) {
            for (u32 m = 1; m <= 6; m++) {
              for (u32 n = 1; n <= 8; n++) {
                GemmMicrokernelTester()
                    .mr(6)
                    .nr(8)
                    .np(8)
                    .kr(1)
                    .m(m)
                    .n(n)
                    .k(k)
                    .iterations(3)
                    .test(pytorch_sgemm_ukernel_6x8__neon);
              }
            }
          }

        */
    }
}

#[test] fn sgemm_6x8_psimd_k_eq_2() {
    todo!();
    /*
    
      GemmMicrokernelTester().mr(6).nr(8).np(8).kr(1).m(6).n(8).k(2).test(
          pytorch_sgemm_ukernel_6x8__psimd);

    */
}

#[test] fn sgemm_6x8_psimd_k_eq_2_strided_a() {
    todo!();
    /*
    
      GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .np(8)
          .kr(1)
          .m(6)
          .n(8)
          .k(2)
          .aStride(37)
          .test(pytorch_sgemm_ukernel_6x8__psimd);

    */
}

#[test] fn sgemm_6x8_psimd_k_eq_2_strided_c() {
    todo!();
    /*
    
      GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .np(8)
          .kr(1)
          .m(6)
          .n(8)
          .k(2)
          .cStride(17)
          .test(pytorch_sgemm_ukernel_6x8__psimd);

    */
}

#[test] fn sgemm_6x8_psimd_k_eq_8_qmin128() {
    todo!();
    /*
    
      GemmMicrokernelTester().mr(6).nr(8).np(8).kr(1).m(6).n(8).k(8).qmin(128).test(
          pytorch_sgemm_ukernel_6x8__psimd);

    */
}

#[test] fn sgemm_6x8_psimd_k_eq_8_qmax128() {
    todo!();
    /*
    
      GemmMicrokernelTester().mr(6).nr(8).np(8).kr(1).m(6).n(8).k(8).qmax(128).test(
          pytorch_sgemm_ukernel_6x8__psimd);

    */
}

#[test] fn sgemm_6x8_psimd_k_gt_2() {
    todo!();
    /*
    
      for (usize k = 3; k < 16; k++) {
        GemmMicrokernelTester().mr(6).nr(8).np(8).kr(1).m(6).n(8).k(k).test(
            pytorch_sgemm_ukernel_6x8__psimd);
      }

    */
}

#[test] fn sgemm_6x8_psimd_k_gt_2_strided_a() {
    todo!();
    /*
    
      for (usize k = 3; k < 16; k++) {
        GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .np(8)
            .kr(1)
            .m(6)
            .n(8)
            .k(k)
            .aStride(37)
            .test(pytorch_sgemm_ukernel_6x8__psimd);
      }

    */
}

#[test] fn sgemm_6x8_psimd_k_gt_2_strided_c() {
    todo!();
    /*
    
      for (usize k = 3; k < 16; k++) {
        GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .np(8)
            .kr(1)
            .m(6)
            .n(8)
            .k(k)
            .cStride(17)
            .test(pytorch_sgemm_ukernel_6x8__psimd);
      }

    */
}

#[test] fn sgemm_6x8_psimd_k_gt_2_subtile() {
    todo!();
    /*
    
      for (usize k = 3; k < 16; k++) {
        for (u32 m = 1; m <= 6; m++) {
          for (u32 n = 1; n <= 8; n++) {
            GemmMicrokernelTester()
                .mr(6)
                .nr(8)
                .np(8)
                .kr(1)
                .m(m)
                .n(n)
                .k(k)
                .iterations(3)
                .test(pytorch_sgemm_ukernel_6x8__psimd);
          }
        }
      }

    */
}

#[test] fn sgemm_6x8_psimd_k_div_2() {
    todo!();
    /*
    
      for (usize k = 2; k < 32; k += 2) {
        GemmMicrokernelTester().mr(6).nr(8).np(8).kr(1).m(6).n(8).k(k).test(
            pytorch_sgemm_ukernel_6x8__psimd);
      }

    */
}

#[test] fn sgemm_6x8_psimd_k_div_2_strided_a() {
    todo!();
    /*
    
      for (usize k = 2; k < 32; k += 2) {
        GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .np(8)
            .kr(1)
            .m(6)
            .n(8)
            .k(k)
            .aStride(171)
            .test(pytorch_sgemm_ukernel_6x8__psimd);
      }

    */
}

#[test] fn sgemm_6x8_psimd_k_div_2_strided_c() {
    todo!();
    /*
    
      for (usize k = 2; k < 32; k += 2) {
        GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .np(8)
            .kr(1)
            .m(6)
            .n(8)
            .k(k)
            .cStride(17)
            .test(pytorch_sgemm_ukernel_6x8__psimd);
      }

    */
}

#[test] fn sgemm_6x8_psimd_k_div_2_subtile() {
    todo!();
    /*
    
      for (usize k = 2; k < 32; k += 6) {
        for (u32 m = 1; m <= 6; m++) {
          for (u32 n = 1; n <= 8; n++) {
            GemmMicrokernelTester()
                .mr(6)
                .nr(8)
                .np(8)
                .kr(1)
                .m(m)
                .n(n)
                .k(k)
                .iterations(3)
                .test(pytorch_sgemm_ukernel_6x8__psimd);
          }
        }
      }

    */
}
