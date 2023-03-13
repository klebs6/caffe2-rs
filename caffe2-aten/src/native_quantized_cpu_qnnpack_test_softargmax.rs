crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/softargmax.cc]

#[test] fn softargmax_op_zero_batch() {
    todo!();
    /*
    
      SoftArgMaxOperatorTester().batchSize(0).channels(1).iterations(1).testQ8();

    */
}

#[test] fn softargmax_op_single_class() {
    todo!();
    /*
    
      SoftArgMaxOperatorTester().batchSize(1).channels(1).iterations(100).testQ8();

    */
}

#[test] fn softargmax_op_two_classes() {
    todo!();
    /*
    
      SoftArgMaxOperatorTester().batchSize(1).channels(2).iterations(100).testQ8();

    */
}

#[test] fn softargmax_op_many_classes() {
    todo!();
    /*
    
      for (usize channels = 3; channels < 100; channels++) {
        SoftArgMaxOperatorTester()
            .batchSize(1)
            .channels(channels)
            .iterations(1)
            .testQ8();
      }

    */
}

#[test] fn softargmax_op_cifar_classes() {
    todo!();
    /*
    
      /* CIFAR-10 */
      SoftArgMaxOperatorTester().batchSize(1).channels(10).iterations(15).testQ8();
      /* CIFAR-100 */
      SoftArgMaxOperatorTester().batchSize(1).channels(100).iterations(15).testQ8();

    */
}

#[test] fn softargmax_op_imagenet_classes() {
    todo!();
    /*
    
      /* ImageNet-1K */
      SoftArgMaxOperatorTester()
          .batchSize(1)
          .channels(1000)
          .iterations(10)
          .testQ8();
      /* ImageNet-1K+1 */
      SoftArgMaxOperatorTester()
          .batchSize(1)
          .channels(1001)
          .iterations(10)
          .testQ8();
      /* ImageNet-22K */
      SoftArgMaxOperatorTester()
          .batchSize(1)
          .channels(21841)
          .iterations(10)
          .testQ8();

    */
}

#[test] fn softargmax_op_many_channels_with_input_scale() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 5) {
        for (float inputScale = 1.0e-2f; inputScale < 1.0e+2f;
             inputScale *= 3.14159265f) {
          SoftArgMaxOperatorTester()
              .batchSize(1)
              .channels(channels)
              .inputScale(inputScale)
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn softargmax_op_many_channels_with_input_zero_point() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 5) {
        for (i32 inputZeroPoint = 0; inputZeroPoint <= 255;
             inputZeroPoint += 51) {
          SoftArgMaxOperatorTester()
              .batchSize(1)
              .channels(channels)
              .inputZeroPoint(u8(inputZeroPoint))
              .iterations(1)
              .testQ8();
        }
      }

    */
}

#[test] fn softargmax_op_small_batch() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 5) {
        SoftArgMaxOperatorTester()
            .batchSize(3)
            .channels(channels)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn softargmax_op_small_batch_with_input_stride() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 5) {
        SoftArgMaxOperatorTester()
            .batchSize(3)
            .channels(channels)
            .inputStride(129)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn softargmax_op_small_batch_with_output_stride() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 5) {
        SoftArgMaxOperatorTester()
            .batchSize(3)
            .channels(channels)
            .outputStride(117)
            .iterations(3)
            .testQ8();
      }

    */
}

#[test] fn softargmax_op_strided_batch_with_input_and_output_stride() {
    todo!();
    /*
    
      for (usize channels = 1; channels < 100; channels += 5) {
        SoftArgMaxOperatorTester()
            .batchSize(3)
            .channels(channels)
            .inputStride(129)
            .outputStride(117)
            .iterations(3)
            .testQ8();
      }

    */
}
