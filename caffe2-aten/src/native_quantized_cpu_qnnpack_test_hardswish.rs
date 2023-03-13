// # vim: ft=none
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/hardswish.cc]

#[test] fn hardswish_op_zero_batch() {
    todo!();
    /*
    
      HardswishOperatorTester().batchSize(0).channels(8).iterations(1).testQ8();

    */
}

#[test] fn hardswish_op_unit_batch() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        HardswishOperatorTester()
            .batchSize(1)
            .channels(channels)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn hardswish_op_unit_batch_with_qmin() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        HardswishOperatorTester()
            .batchSize(1)
            .channels(channels)
            .qmin(128)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn hardswish_op_unit_batch_with_qmax() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        HardswishOperatorTester()
            .batchSize(1)
            .channels(channels)
            .qmax(128)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn hardswish_op_unit_batch_with_input_scale() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (float inputScale = 1.0e-2f; inputScale < 1.0e+2f;
             inputScale *= 10.0f) {
          HardswishOperatorTester()
              .batchSize(1)
              .channels(channels)
              .inputScale(inputScale)
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn hardswish_op_unit_batch_with_input_zero_point() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (i32 inputZeroPoint = 0; inputZeroPoint <= 255;
             inputZeroPoint += 51) {
          HardswishOperatorTester()
              .batchSize(1)
              .channels(channels)
              .inputZeroPoint(u8(inputZeroPoint))
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn hardswish_op_unit_batch_with_output_scale() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (float outputScale = 1.0e-2f; outputScale < 1.0e+2f;
             outputScale *= 10.0f) {
          HardswishOperatorTester()
              .batchSize(1)
              .channels(channels)
              .outputScale(outputScale)
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn hardswish_op_unit_batch_with_output_zero_point() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (i32 outputZeroPoint = 0; outputZeroPoint <= 255;
             outputZeroPoint += 51) {
          HardswishOperatorTester()
              .batchSize(1)
              .channels(channels)
              .outputZeroPoint(u8(outputZeroPoint))
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn hardswish_op_small_batch() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        HardswishOperatorTester()
            .batchSize(3)
            .channels(channels)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn hardswish_op_small_batch_with_input_stride() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        HardswishOperatorTester()
            .batchSize(3)
            .channels(channels)
            .inputStride(129)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn hardswish_op_small_batch_with_output_stride() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        HardswishOperatorTester()
            .batchSize(3)
            .channels(channels)
            .outputStride(117)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn hardswish_op_small_batch_with_qmin() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        HardswishOperatorTester()
            .batchSize(3)
            .channels(channels)
            .qmin(128)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn hardswish_op_small_batch_with_qmax() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        HardswishOperatorTester()
            .batchSize(3)
            .channels(channels)
            .qmax(128)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn hardswish_op_small_batch_with_input_scale() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (float inputScale = 1.0e-2f; inputScale < 1.0e+2f;
             inputScale *= 10.0f) {
          HardswishOperatorTester()
              .batchSize(3)
              .channels(channels)
              .inputScale(inputScale)
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn hardswish_op_small_batch_with_input_zero_point() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (i32 inputZeroPoint = 0; inputZeroPoint <= 255;
             inputZeroPoint += 51) {
          HardswishOperatorTester()
              .batchSize(3)
              .channels(channels)
              .inputZeroPoint(u8(inputZeroPoint))
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn hardswish_op_strided_batch() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        HardswishOperatorTester()
            .batchSize(3)
            .channels(channels)
            .inputStride(129)
            .outputStride(117)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn hardswish_op_strided_batch_with_qmin() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        HardswishOperatorTester()
            .batchSize(3)
            .channels(channels)
            .inputStride(129)
            .outputStride(117)
            .qmin(128)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn hardswish_op_strided_batch_with_qmax() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        HardswishOperatorTester()
            .batchSize(3)
            .channels(channels)
            .inputStride(129)
            .outputStride(117)
            .qmax(128)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn hardswish_op_strided_batch_with_input_scale() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (float inputScale = 1.0e-2f; inputScale < 1.0e+2f;
             inputScale *= 10.0f) {
          HardswishOperatorTester()
              .batchSize(3)
              .channels(channels)
              .inputStride(129)
              .outputStride(117)
              .inputScale(inputScale)
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn hardswish_op_strided_batch_with_input_zero_point() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (i32 inputZeroPoint = 0; inputZeroPoint <= 255;
             inputZeroPoint += 51) {
          HardswishOperatorTester()
              .batchSize(3)
              .channels(channels)
              .inputStride(129)
              .outputStride(117)
              .inputZeroPoint(u8(inputZeroPoint))
              .iterations(1)
              .testQ8();
        }
      }

    */
}


