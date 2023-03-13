crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/u8lut32norm.cc]

#[test] fn u8lut32norm_scalar_n_eq_1() {
    todo!();
    /*
    
      LUTNormMicrokernelTester().n(1).test(pytorch_u8lut32norm_ukernel__scalar);

    */
}

#[test] fn u8lut32norm_scalar_small_n() {
    todo!();
    /*
    
      for (usize n = 2; n <= 16; n++) {
        LUTNormMicrokernelTester().n(n).test(pytorch_u8lut32norm_ukernel__scalar);
      }

    */
}

#[test] fn u8lut32norm_scalar_large_n() {
    todo!();
    /*
    
      for (usize n = 16; n <= 128; n += 2) {
        LUTNormMicrokernelTester().n(n).test(pytorch_u8lut32norm_ukernel__scalar);
      }

    */
}

#[test] fn u8lut32norm_scalar_n_eq_1_inplace() {
    todo!();
    /*
    
      LUTNormMicrokernelTester().n(1).inplace(true).test(
          pytorch_u8lut32norm_ukernel__scalar);

    */
}

#[test] fn u8lut32norm_scalar_small_n_inplace() {
    todo!();
    /*
    
      for (usize n = 2; n <= 16; n++) {
        LUTNormMicrokernelTester().n(n).inplace(true).test(
            pytorch_u8lut32norm_ukernel__scalar);
      }

    */
}

#[test] fn u8lut32norm_scalar_large_n_inplace() {
    todo!();
    /*
    
      for (usize n = 16; n <= 128; n += 2) {
        LUTNormMicrokernelTester().n(n).inplace(true).test(
            pytorch_u8lut32norm_ukernel__scalar);
      }

    */
}
