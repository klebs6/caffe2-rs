crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/clamp.cc]

#[test] fn clamp_op_zero_batch() {
    todo!();
    /*
    
      ClampOperatorTester().batchSize(0).channels(2).iterations(1).testU8();

    */
}

#[test] fn clamp_op_unit_batch() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels++) {
        ClampOperatorTester()
            .batchSize(1)
            .channels(channels)
            .iterations(3)
            .testU8();
      }

    */
}

#[test] fn clamp_op_unit_batch_with_qmin() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (u8 qmin = 1; qmin < 255; qmin++) {
          ClampOperatorTester()
              .batchSize(1)
              .channels(channels)
              .qmin(qmin)
              .iterations(3)
              .testU8();
        }
      }

    */
}

#[test] fn clamp_op_unit_batch_with_qmax() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (u8 qmax = 1; qmax < 255; qmax++) {
          ClampOperatorTester()
              .batchSize(1)
              .channels(channels)
              .qmax(qmax)
              .iterations(3)
              .testU8();
        }
      }

    */
}

#[test] fn clamp_op_small_batch() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels++) {
        ClampOperatorTester()
            .batchSize(3)
            .channels(channels)
            .iterations(3)
            .testU8();
      }

    */
}

#[test] fn clamp_op_small_batch_with_input_stride() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        ClampOperatorTester()
            .batchSize(3)
            .channels(channels)
            .inputStride(129)
            .iterations(3)
            .testU8();
      }

    */
}

#[test] fn clamp_op_small_batch_with_output_stride() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        ClampOperatorTester()
            .batchSize(3)
            .channels(channels)
            .outputStride(117)
            .iterations(3)
            .testU8();
      }

    */
}

#[test] fn clamp_op_small_batch_with_input_and_output_stride() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        ClampOperatorTester()
            .batchSize(3)
            .channels(channels)
            .inputStride(129)
            .outputStride(117)
            .iterations(3)
            .testU8();
      }

    */
}

#[test] fn clamp_op_qmin_and_qmax_equal_uint8_max() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        ClampOperatorTester()
            .batchSize(3)
            .channels(channels)
            .qmin(255)
            .qmax(255)
            .iterations(3)
            .testU8();
      }

    */
}
