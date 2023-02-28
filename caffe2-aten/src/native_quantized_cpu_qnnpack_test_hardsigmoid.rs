// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/hardsigmoid.cc]

#[test] fn HARDSIGMOID_OP_zero_batch() {
    todo!();
    /*
    
      HardsigmoidOperatorTester().batchSize(0).channels(8).iterations(1).testQ8();

    */
}

#[test] fn HARDSIGMOID_OP_unit_batch() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        HardsigmoidOperatorTester()
            .batchSize(1)
            .channels(channels)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn HARDSIGMOID_OP_unit_batch_with_qmin() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        HardsigmoidOperatorTester()
            .batchSize(1)
            .channels(channels)
            .qmin(128)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn HARDSIGMOID_OP_unit_batch_with_qmax() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        HardsigmoidOperatorTester()
            .batchSize(1)
            .channels(channels)
            .qmax(128)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn HARDSIGMOID_OP_unit_batch_with_input_scale() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (float inputScale = 1.0e-2f; inputScale < 1.0e+2f;
             inputScale *= 10.0f) {
          HardsigmoidOperatorTester()
              .batchSize(1)
              .channels(channels)
              .inputScale(inputScale)
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn HARDSIGMOID_OP_unit_batch_with_input_zero_point() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (i32 inputZeroPoint = 0; inputZeroPoint <= 255;
             inputZeroPoint += 51) {
          HardsigmoidOperatorTester()
              .batchSize(1)
              .channels(channels)
              .inputZeroPoint(u8(inputZeroPoint))
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn HARDSIGMOID_OP_small_batch() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        HardsigmoidOperatorTester()
            .batchSize(3)
            .channels(channels)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn HARDSIGMOID_OP_small_batch_with_input_stride() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        HardsigmoidOperatorTester()
            .batchSize(3)
            .channels(channels)
            .inputStride(129)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn HARDSIGMOID_OP_small_batch_with_output_stride() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        HardsigmoidOperatorTester()
            .batchSize(3)
            .channels(channels)
            .outputStride(117)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn HARDSIGMOID_OP_small_batch_with_qmin() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        HardsigmoidOperatorTester()
            .batchSize(3)
            .channels(channels)
            .qmin(128)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn HARDSIGMOID_OP_small_batch_with_qmax() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        HardsigmoidOperatorTester()
            .batchSize(3)
            .channels(channels)
            .qmax(128)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn HARDSIGMOID_OP_small_batch_with_input_scale() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (float inputScale = 1.0e-2f; inputScale < 1.0e+2f;
             inputScale *= 10.0f) {
          HardsigmoidOperatorTester()
              .batchSize(3)
              .channels(channels)
              .inputScale(inputScale)
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn HARDSIGMOID_OP_small_batch_with_input_zero_point() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (i32 inputZeroPoint = 0; inputZeroPoint <= 255;
             inputZeroPoint += 51) {
          HardsigmoidOperatorTester()
              .batchSize(3)
              .channels(channels)
              .inputZeroPoint(u8(inputZeroPoint))
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn HARDSIGMOID_OP_strided_batch() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        HardsigmoidOperatorTester()
            .batchSize(3)
            .channels(channels)
            .inputStride(129)
            .outputStride(117)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn HARDSIGMOID_OP_strided_batch_with_qmin() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        HardsigmoidOperatorTester()
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

#[test] fn HARDSIGMOID_OP_strided_batch_with_qmax() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        HardsigmoidOperatorTester()
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

#[test] fn HARDSIGMOID_OP_strided_batch_with_input_scale() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (float inputScale = 1.0e-2f; inputScale < 1.0e+2f;
             inputScale *= 10.0f) {
          HardsigmoidOperatorTester()
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

#[test] fn HARDSIGMOID_OP_strided_batch_with_input_zero_point() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (i32 inputZeroPoint = 0; inputZeroPoint <= 255;
             inputZeroPoint += 51) {
          HardsigmoidOperatorTester()
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


