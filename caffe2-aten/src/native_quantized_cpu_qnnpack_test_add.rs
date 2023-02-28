crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/add.cc]

#[test] fn ADD_OP_zero_batch() {
    todo!();
    /*
    
      AddOperatorTester().batchSize(0).channels(2).iterations(1).testQ8();

    */
}

#[test] fn ADD_OP_unit_batch() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        AddOperatorTester().batchSize(1).channels(channels).iterations(3).testQ8();
      }

    */
}

#[test] fn ADD_OP_unit_batch_with_qmin() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        AddOperatorTester()
            .batchSize(1)
            .channels(channels)
            .qmin(128)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn ADD_OP_unit_batch_with_qmax() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        AddOperatorTester()
            .batchSize(1)
            .channels(channels)
            .qmax(128)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn ADD_OP_unit_batch_with_a_scale() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (float aScale = 1.0e-2f; aScale < 1.0e+2f; aScale *= 10.0f) {
          AddOperatorTester()
              .batchSize(1)
              .channels(channels)
              .aScale(aScale)
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn ADD_OP_unit_batch_with_b_scale() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (float bScale = 1.0e-2f; bScale < 1.0e+2f; bScale *= 10.0f) {
          AddOperatorTester()
              .batchSize(1)
              .channels(channels)
              .bScale(bScale)
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn ADD_OP_unit_batch_with_y_scale() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (float yScale = 1.0e-2f; yScale < 1.0e+2f; yScale *= 10.0f) {
          AddOperatorTester()
              .batchSize(1)
              .channels(channels)
              .yScale(yScale)
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn ADD_OP_unit_batch_with_a_zero_point() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (i32 aZeroPoint = 0; aZeroPoint <= 255; aZeroPoint += 51) {
          AddOperatorTester()
              .batchSize(1)
              .channels(channels)
              .aZeroPoint(u8(aZeroPoint))
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn ADD_OP_unit_batch_with_b_zero_point() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (i32 bZeroPoint = 0; bZeroPoint <= 255; bZeroPoint += 51) {
          AddOperatorTester()
              .batchSize(1)
              .channels(channels)
              .bZeroPoint(u8(bZeroPoint))
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn ADD_OP_unit_batch_with_y_zero_point() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (i32 yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
          AddOperatorTester()
              .batchSize(1)
              .channels(channels)
              .yZeroPoint(u8(yZeroPoint))
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn ADD_OP_small_batch() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        AddOperatorTester().batchSize(3).channels(channels).iterations(3).testQ8();
      }

    */
}

#[test] fn ADD_OP_small_batch_with_a_stride() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        AddOperatorTester()
            .batchSize(3)
            .channels(channels)
            .aStride(129)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn ADD_OP_small_batch_with_b_stride() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        AddOperatorTester()
            .batchSize(3)
            .channels(channels)
            .bStride(123)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn ADD_OP_small_batch_with_y_stride() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        AddOperatorTester()
            .batchSize(3)
            .channels(channels)
            .yStride(117)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn ADD_OP_small_batch_with_qmin() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        AddOperatorTester()
            .batchSize(3)
            .channels(channels)
            .qmin(128)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn ADD_OP_small_batch_with_qmax() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        AddOperatorTester()
            .batchSize(3)
            .channels(channels)
            .qmax(128)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn ADD_OP_small_batch_with_a_scale() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (float aScale = 1.0e-2f; aScale < 1.0e+2f; aScale *= 10.0f) {
          AddOperatorTester()
              .batchSize(3)
              .channels(channels)
              .aScale(aScale)
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn ADD_OP_small_batch_with_b_scale() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (float bScale = 1.0e-2f; bScale < 1.0e+2f; bScale *= 10.0f) {
          AddOperatorTester()
              .batchSize(3)
              .channels(channels)
              .bScale(bScale)
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn ADD_OP_small_batch_with_y_scale() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (float yScale = 1.0e-2f; yScale < 1.0e+2f; yScale *= 10.0f) {
          AddOperatorTester()
              .batchSize(3)
              .channels(channels)
              .yScale(yScale)
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn ADD_OP_small_batch_with_a_zero_point() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (i32 aZeroPoint = 0; aZeroPoint <= 255; aZeroPoint += 51) {
          AddOperatorTester()
              .batchSize(3)
              .channels(channels)
              .aZeroPoint(u8(aZeroPoint))
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn ADD_OP_small_batch_with_b_zero_point() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (i32 bZeroPoint = 0; bZeroPoint <= 255; bZeroPoint += 51) {
          AddOperatorTester()
              .batchSize(3)
              .channels(channels)
              .bZeroPoint(u8(bZeroPoint))
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn ADD_OP_small_batch_with_y_zero_point() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (i32 yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
          AddOperatorTester()
              .batchSize(3)
              .channels(channels)
              .yZeroPoint(u8(yZeroPoint))
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn ADD_OP_strided_batch() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        AddOperatorTester()
            .batchSize(3)
            .channels(channels)
            .aStride(129)
            .bStride(123)
            .yStride(117)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn ADD_OP_strided_batch_with_qmin() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        AddOperatorTester()
            .batchSize(3)
            .channels(channels)
            .aStride(129)
            .bStride(123)
            .yStride(117)
            .qmin(128)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn ADD_OP_strided_batch_with_qmax() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        AddOperatorTester()
            .batchSize(3)
            .channels(channels)
            .aStride(129)
            .bStride(123)
            .yStride(117)
            .qmax(128)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn ADD_OP_strided_batch_with_a_scale() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (float aScale = 1.0e-2f; aScale < 1.0e+2f; aScale *= 10.0f) {
          AddOperatorTester()
              .batchSize(3)
              .channels(channels)
              .aStride(129)
              .bStride(123)
              .yStride(117)
              .aScale(aScale)
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn ADD_OP_strided_batch_with_b_scale() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (float bScale = 1.0e-2f; bScale < 1.0e+2f; bScale *= 10.0f) {
          AddOperatorTester()
              .batchSize(3)
              .channels(channels)
              .aStride(129)
              .bStride(123)
              .yStride(117)
              .bScale(bScale)
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn ADD_OP_strided_batch_with_y_scale() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (float yScale = 1.0e-2f; yScale < 1.0e+2f; yScale *= 10.0f) {
          AddOperatorTester()
              .batchSize(3)
              .channels(channels)
              .aStride(129)
              .bStride(123)
              .yStride(117)
              .yScale(yScale)
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn ADD_OP_strided_batch_with_a_zero_point() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (i32 aZeroPoint = 0; aZeroPoint <= 255; aZeroPoint += 51) {
          AddOperatorTester()
              .batchSize(3)
              .channels(channels)
              .aStride(129)
              .bStride(123)
              .yStride(117)
              .aZeroPoint(u8(aZeroPoint))
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn ADD_OP_strided_batch_with_b_zero_point() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (i32 bZeroPoint = 0; bZeroPoint <= 255; bZeroPoint += 51) {
          AddOperatorTester()
              .batchSize(3)
              .channels(channels)
              .aStride(129)
              .bStride(123)
              .yStride(117)
              .bZeroPoint(u8(bZeroPoint))
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn ADD_OP_strided_batch_with_y_zero_point() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 15) {
        for (i32 yZeroPoint = 0; yZeroPoint <= 255; yZeroPoint += 51) {
          AddOperatorTester()
              .batchSize(3)
              .channels(channels)
              .aStride(129)
              .bStride(123)
              .yStride(117)
              .yZeroPoint(u8(yZeroPoint))
              .iterations(1)
              .testQ8();
        }
      }

    */
}
