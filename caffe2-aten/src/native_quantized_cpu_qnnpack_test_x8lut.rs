crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/x8lut.cc]

#[test] fn x8lut_scalar_n_eq_1() {
    todo!();
    /*
    
      LUTMicrokernelTester().n(1).test(pytorch_x8lut_ukernel__scalar);

    */
}

#[test] fn x8lut_scalar_small_n() {
    todo!();
    /*
    
      for (usize n = 2; n <= 16; n++) {
        LUTMicrokernelTester().n(n).test(pytorch_x8lut_ukernel__scalar);
      }

    */
}

#[test] fn x8lut_scalar_large_n() {
    todo!();
    /*
    
      for (usize n = 16; n <= 128; n += 2) {
        LUTMicrokernelTester().n(n).test(pytorch_x8lut_ukernel__scalar);
      }

    */
}

#[test] fn x8lut_scalar_n_eq_1_inplace() {
    todo!();
    /*
    
      LUTMicrokernelTester().n(1).inplace(true).test(pytorch_x8lut_ukernel__scalar);

    */
}

#[test] fn x8lut_scalar_small_n_inplace() {
    todo!();
    /*
    
      for (usize n = 2; n <= 16; n++) {
        LUTMicrokernelTester().n(n).inplace(true).test(pytorch_x8lut_ukernel__scalar);
      }

    */
}

#[test] fn x8lut_scalar_large_n_inplace() {
    todo!();
    /*
    
      for (usize n = 16; n <= 128; n += 2) {
        LUTMicrokernelTester().n(n).inplace(true).test(pytorch_x8lut_ukernel__scalar);
      }

    */
}
