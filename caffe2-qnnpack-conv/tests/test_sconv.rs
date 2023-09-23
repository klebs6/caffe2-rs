crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/sconv.cc]

#[test] fn sconv_6x8_psimd_k_eq_1() {
    todo!();
    /*
    
      GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .np(8)
          .kr(1)
          .m(6)
          .n(8)
          .k(1)
          .aStride(37)
          .test(pytorch_sconv_ukernel_6x8__psimd);

    */
}

#[test] fn sconv_6x8_psimd_k_eq_1_strided_c() {
    todo!();
    /*
    
      GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .np(8)
          .kr(1)
          .m(6)
          .n(8)
          .k(1)
          .aStride(37)
          .cStride(17)
          .test(pytorch_sconv_ukernel_6x8__psimd);

    */
}

#[test] fn sconv_6x8_psimd_k_eq_1_qmin128() {
    todo!();
    /*
    
      GemmMicrokernelTester().mr(6).nr(8).np(8).kr(1).m(6).n(8).k(1).qmin(128).test(
          pytorch_sconv_ukernel_6x8__psimd);

    */
}

#[test] fn sconv_6x8_psimd_k_eq_1_qmax128() {
    todo!();
    /*
    
      GemmMicrokernelTester().mr(6).nr(8).np(8).kr(1).m(6).n(8).k(1).qmax(128).test(
          pytorch_sconv_ukernel_6x8__psimd);

    */
}

#[test] fn sconv_6x8_psimd_k_gt_1() {
    todo!();
    /*
    
      for (usize k = 2; k < 16; k++) {
        GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .np(8)
            .kr(1)
            .m(6)
            .n(8)
            .k(k)
            .aStride(37)
            .test(pytorch_sconv_ukernel_6x8__psimd);
      }

    */
}

#[test] fn sconv_6x8_psimd_k_gt_1_strided_c() {
    todo!();
    /*
    
      for (usize k = 2; k < 16; k++) {
        GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .np(8)
            .kr(1)
            .m(6)
            .n(8)
            .k(k)
            .aStride(37)
            .cStride(17)
            .test(pytorch_sconv_ukernel_6x8__psimd);
      }

    */
}

#[test] fn sconv_6x8_psimd_k_gt_1_subtile() {
    todo!();
    /*
    
      for (usize k = 2; k < 16; k++) {
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
                .aStride(37)
                .iterations(3)
                .test(pytorch_sconv_ukernel_6x8__psimd);
          }
        }
      }

    */
}
