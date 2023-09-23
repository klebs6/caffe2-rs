crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/average-pooling.cc]

#[test] fn average_pooling_op_zero_batch() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      AveragePoolingOperatorTester()
          .batchSize(0)
          .inputHeight(2)
          .inputWidth(4)
          .poolingHeight(1)
          .poolingWidth(2)
          .channels(4)
          .testQ8();

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_small_1xm_pool() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_small_1xm_pool_with_padding() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels += 3) {
        for (usize poolSize = 3; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
             poolSize++) {
          for (usize paddingLeft = 0; paddingLeft <= 1; paddingLeft++) {
            for (usize paddingRight = 0; paddingRight <= 1; paddingRight++) {
              AveragePoolingOperatorTester()
                  .batchSize(1)
                  .inputHeight(2)
                  .inputWidth(poolSize + 2)
                  .paddingLeft(paddingLeft)
                  .paddingRight(paddingRight)
                  .poolingHeight(1)
                  .poolingWidth(poolSize)
                  .channels(channels)
                  .testQ8();
            }
          }
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_small_1xm_pool_with_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels += 3) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 4)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .strideWidth(2)
              .channels(channels)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_small_mx1_pool() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_small_mx1_pool_with_padding() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels += 3) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
             poolSize++) {
          for (usize paddingTop = 0; paddingTop <= 1; paddingTop++) {
            for (usize paddingBottom = 0; paddingBottom <= 1; paddingBottom++) {
              AveragePoolingOperatorTester()
                  .batchSize(1)
                  .inputHeight(poolSize + 1)
                  .inputWidth(3)
                  .paddingTop(paddingTop)
                  .paddingBottom(paddingBottom)
                  .poolingHeight(poolSize)
                  .poolingWidth(1)
                  .channels(channels)
                  .testQ8();
            }
          }
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_small_mx1_pool_with_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels += 3) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 3)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .strideHeight(2)
              .channels(channels)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_small_pool_with_input_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels += 3) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .inputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
              .testQ8();
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .inputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_small_pool_with_output_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels += 3) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .outputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
              .testQ8();
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .outputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_small_pool_with_input_scale() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels += 3) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
             poolSize++) {
          for (float inputScale = 0.01f; inputScale < 100.0f;
               inputScale *= 3.14159265f) {
            AveragePoolingOperatorTester()
                .batchSize(1)
                .inputHeight(poolSize + 1)
                .inputWidth(3)
                .poolingHeight(poolSize)
                .poolingWidth(1)
                .channels(channels)
                .inputScale(inputScale)
                .testQ8();
            AveragePoolingOperatorTester()
                .batchSize(1)
                .inputHeight(2)
                .inputWidth(poolSize + 2)
                .poolingHeight(1)
                .poolingWidth(poolSize)
                .channels(channels)
                .inputScale(inputScale)
                .testQ8();
          }
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_small_pool_with_input_zero_point() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels += 3) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
             poolSize++) {
          for (i32 inputZeroPoint = 0; inputZeroPoint <= 255;
               inputZeroPoint += 51) {
            AveragePoolingOperatorTester()
                .batchSize(1)
                .inputHeight(poolSize + 1)
                .inputWidth(3)
                .poolingHeight(poolSize)
                .poolingWidth(1)
                .channels(channels)
                .inputZeroPoint(u8(inputZeroPoint))
                .testQ8();
            AveragePoolingOperatorTester()
                .batchSize(1)
                .inputHeight(2)
                .inputWidth(poolSize + 2)
                .poolingHeight(1)
                .poolingWidth(poolSize)
                .channels(channels)
                .inputZeroPoint(u8(inputZeroPoint))
                .testQ8();
          }
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_small_pool_with_output_scale() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels += 3) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
             poolSize++) {
          for (float outputScale = 0.01f; outputScale < 100.0f;
               outputScale *= 3.14159265f) {
            AveragePoolingOperatorTester()
                .batchSize(1)
                .inputHeight(poolSize + 1)
                .inputWidth(3)
                .poolingHeight(poolSize)
                .poolingWidth(1)
                .channels(channels)
                .outputScale(outputScale)
                .testQ8();
            AveragePoolingOperatorTester()
                .batchSize(1)
                .inputHeight(2)
                .inputWidth(poolSize + 2)
                .poolingHeight(1)
                .poolingWidth(poolSize)
                .channels(channels)
                .outputScale(outputScale)
                .testQ8();
          }
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_small_pool_with_output_zero_point() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels += 3) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
             poolSize++) {
          for (i32 outputZeroPoint = 0; outputZeroPoint <= 255;
               outputZeroPoint += 51) {
            AveragePoolingOperatorTester()
                .batchSize(1)
                .inputHeight(poolSize + 1)
                .inputWidth(3)
                .poolingHeight(poolSize)
                .poolingWidth(1)
                .channels(channels)
                .outputZeroPoint(u8(outputZeroPoint))
                .testQ8();
            AveragePoolingOperatorTester()
                .batchSize(1)
                .inputHeight(2)
                .inputWidth(poolSize + 2)
                .poolingHeight(1)
                .poolingWidth(poolSize)
                .channels(channels)
                .outputZeroPoint(u8(outputZeroPoint))
                .testQ8();
          }
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_small_pool_with_qmin() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels += 3) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .qmin(128)
              .testQ8();
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .qmin(128)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_small_pool_with_qmax() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels += 3) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .qmax(128)
              .testQ8();
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .qmax(128)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_large_1xm_pool() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels++) {
        for (usize poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
             pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_large_1xm_pool_with_padding() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels += 3) {
        for (usize poolSize = 3; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
             poolSize++) {
          for (usize paddingLeft = 0; paddingLeft <= 1; paddingLeft++) {
            for (usize paddingRight = 0; paddingRight <= 1; paddingRight++) {
              AveragePoolingOperatorTester()
                  .batchSize(1)
                  .inputHeight(2)
                  .inputWidth(poolSize + 2)
                  .paddingLeft(paddingLeft)
                  .paddingRight(paddingRight)
                  .poolingHeight(1)
                  .poolingWidth(poolSize)
                  .channels(channels)
                  .testQ8();
            }
          }
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_large_1xm_pool_with_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels++) {
        for (usize poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
             pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 4)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .strideWidth(2)
              .channels(channels)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_large_mx1_pool() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels++) {
        for (usize poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
             pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_large_mx1_pool_with_padding() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels += 3) {
        for (usize poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
             pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
             poolSize++) {
          for (usize paddingTop = 0; paddingTop <= 1; paddingTop++) {
            for (usize paddingBottom = 0; paddingBottom <= 1; paddingBottom++) {
              AveragePoolingOperatorTester()
                  .batchSize(1)
                  .inputHeight(poolSize + 1)
                  .inputWidth(3)
                  .paddingTop(paddingTop)
                  .paddingBottom(paddingBottom)
                  .poolingHeight(poolSize)
                  .poolingWidth(1)
                  .channels(channels)
                  .testQ8();
            }
          }
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_large_mx1_pool_with_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels += 3) {
        for (usize poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
             pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
             poolSize++) {
          for (usize paddingTop = 0; paddingTop <= 1; paddingTop++) {
            for (usize paddingBottom = 0; paddingBottom <= 1; paddingBottom++) {
              AveragePoolingOperatorTester()
                  .batchSize(1)
                  .inputHeight(poolSize + 1)
                  .inputWidth(3)
                  .paddingTop(paddingTop)
                  .paddingBottom(paddingBottom)
                  .poolingHeight(poolSize)
                  .poolingWidth(1)
                  .channels(channels)
                  .testQ8();
            }
          }
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_large_pool_with_input_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels++) {
        for (usize poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
             pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .inputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
              .testQ8();
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .inputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_large_pool_with_input_scale() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels += 3) {
        for (usize poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
             pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
             poolSize++) {
          for (float inputScale = 0.01f; inputScale < 100.0f;
               inputScale *= 3.14159265f) {
            AveragePoolingOperatorTester()
                .batchSize(1)
                .inputHeight(poolSize + 1)
                .inputWidth(3)
                .poolingHeight(poolSize)
                .poolingWidth(1)
                .channels(channels)
                .inputScale(inputScale)
                .testQ8();
            AveragePoolingOperatorTester()
                .batchSize(1)
                .inputHeight(2)
                .inputWidth(poolSize + 2)
                .poolingHeight(1)
                .poolingWidth(poolSize)
                .channels(channels)
                .inputScale(inputScale)
                .testQ8();
          }
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_large_pool_with_input_zero_point() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels += 3) {
        for (usize poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
             pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
             poolSize++) {
          for (i32 inputZeroPoint = 0; inputZeroPoint <= 255;
               inputZeroPoint += 51) {
            AveragePoolingOperatorTester()
                .batchSize(1)
                .inputHeight(poolSize + 1)
                .inputWidth(3)
                .poolingHeight(poolSize)
                .poolingWidth(1)
                .channels(channels)
                .inputZeroPoint(u8(inputZeroPoint))
                .testQ8();
            AveragePoolingOperatorTester()
                .batchSize(1)
                .inputHeight(2)
                .inputWidth(poolSize + 2)
                .poolingHeight(1)
                .poolingWidth(poolSize)
                .channels(channels)
                .inputZeroPoint(u8(inputZeroPoint))
                .testQ8();
          }
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_large_pool_with_output_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels++) {
        for (usize poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
             pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .outputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
              .testQ8();
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .outputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_large_pool_with_output_scale() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels += 3) {
        for (usize poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
             pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
             poolSize++) {
          for (float outputScale = 0.01f; outputScale < 100.0f;
               outputScale *= 3.14159265f) {
            AveragePoolingOperatorTester()
                .batchSize(1)
                .inputHeight(poolSize + 1)
                .inputWidth(3)
                .poolingHeight(poolSize)
                .poolingWidth(1)
                .channels(channels)
                .outputScale(outputScale)
                .testQ8();
            AveragePoolingOperatorTester()
                .batchSize(1)
                .inputHeight(2)
                .inputWidth(poolSize + 2)
                .poolingHeight(1)
                .poolingWidth(poolSize)
                .channels(channels)
                .outputScale(outputScale)
                .testQ8();
          }
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_large_pool_with_output_zero_point() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels += 3) {
        for (usize poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
             pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
             poolSize++) {
          for (i32 outputZeroPoint = 0; outputZeroPoint <= 255;
               outputZeroPoint += 51) {
            AveragePoolingOperatorTester()
                .batchSize(1)
                .inputHeight(poolSize + 1)
                .inputWidth(3)
                .poolingHeight(poolSize)
                .poolingWidth(1)
                .channels(channels)
                .outputZeroPoint(u8(outputZeroPoint))
                .testQ8();
            AveragePoolingOperatorTester()
                .batchSize(1)
                .inputHeight(2)
                .inputWidth(poolSize + 2)
                .poolingHeight(1)
                .poolingWidth(poolSize)
                .channels(channels)
                .outputZeroPoint(u8(outputZeroPoint))
                .testQ8();
          }
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_large_pool_with_qmin() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels += 3) {
        for (usize poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
             pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .qmin(128)
              .testQ8();
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .qmin(128)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_many_channels_large_pool_with_qmax() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels += 3) {
        for (usize poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
             pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .qmax(128)
              .testQ8();
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .qmax(128)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_few_channels_1xm_pool() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_few_channels_1xm_pool_with_padding() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
           channels++) {
        for (usize poolSize = 3; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
             poolSize++) {
          for (usize paddingLeft = 0; paddingLeft <= 1; paddingLeft++) {
            for (usize paddingRight = 0; paddingRight <= 1; paddingRight++) {
              AveragePoolingOperatorTester()
                  .batchSize(1)
                  .inputHeight(2)
                  .inputWidth(poolSize + 2)
                  .paddingLeft(paddingLeft)
                  .paddingRight(paddingRight)
                  .poolingHeight(1)
                  .poolingWidth(poolSize)
                  .channels(channels)
                  .testQ8();
            }
          }
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_few_channels_1xm_pool_with_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 4)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .strideWidth(2)
              .channels(channels)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_few_channels_mx1_pool() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_few_channels_mx1_pool_with_padding() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
             poolSize++) {
          for (usize paddingTop = 0; paddingTop <= 1; paddingTop++) {
            for (usize paddingBottom = 0; paddingBottom <= 1; paddingBottom++) {
              AveragePoolingOperatorTester()
                  .batchSize(1)
                  .inputHeight(poolSize + 1)
                  .inputWidth(3)
                  .paddingTop(paddingTop)
                  .paddingBottom(paddingBottom)
                  .poolingHeight(poolSize)
                  .poolingWidth(1)
                  .channels(channels)
                  .testQ8();
            }
          }
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_few_channels_mx1_pool_with_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 3)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .strideHeight(2)
              .channels(channels)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_few_channels_with_input_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .inputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
              .testQ8();
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .inputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_few_channels_with_input_scale() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
             poolSize++) {
          for (float inputScale = 0.01f; inputScale < 100.0f;
               inputScale *= 3.14159265f) {
            AveragePoolingOperatorTester()
                .batchSize(1)
                .inputHeight(poolSize + 1)
                .inputWidth(3)
                .poolingHeight(poolSize)
                .poolingWidth(1)
                .channels(channels)
                .inputScale(inputScale)
                .testQ8();
            AveragePoolingOperatorTester()
                .batchSize(1)
                .inputHeight(2)
                .inputWidth(poolSize + 2)
                .poolingHeight(1)
                .poolingWidth(poolSize)
                .channels(channels)
                .inputScale(inputScale)
                .testQ8();
          }
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_few_channels_with_input_zero_point() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
             poolSize++) {
          for (i32 inputZeroPoint = 0; inputZeroPoint <= 255;
               inputZeroPoint += 51) {
            AveragePoolingOperatorTester()
                .batchSize(1)
                .inputHeight(poolSize + 1)
                .inputWidth(3)
                .poolingHeight(poolSize)
                .poolingWidth(1)
                .channels(channels)
                .inputZeroPoint(u8(inputZeroPoint))
                .testQ8();
            AveragePoolingOperatorTester()
                .batchSize(1)
                .inputHeight(2)
                .inputWidth(poolSize + 2)
                .poolingHeight(1)
                .poolingWidth(poolSize)
                .channels(channels)
                .inputZeroPoint(u8(inputZeroPoint))
                .testQ8();
          }
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_few_channels_with_output_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .outputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
              .testQ8();
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .outputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_few_channels_with_output_scale() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
             poolSize++) {
          for (float outputScale = 0.01f; outputScale < 100.0f;
               outputScale *= 3.14159265f) {
            AveragePoolingOperatorTester()
                .batchSize(1)
                .inputHeight(poolSize + 1)
                .inputWidth(3)
                .poolingHeight(poolSize)
                .poolingWidth(1)
                .channels(channels)
                .outputScale(outputScale)
                .testQ8();
            AveragePoolingOperatorTester()
                .batchSize(1)
                .inputHeight(2)
                .inputWidth(poolSize + 2)
                .poolingHeight(1)
                .poolingWidth(poolSize)
                .channels(channels)
                .outputScale(outputScale)
                .testQ8();
          }
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_few_channels_with_output_zero_point() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
             poolSize++) {
          for (i32 outputZeroPoint = 0; outputZeroPoint <= 255;
               outputZeroPoint += 51) {
            AveragePoolingOperatorTester()
                .batchSize(1)
                .inputHeight(poolSize + 1)
                .inputWidth(3)
                .poolingHeight(poolSize)
                .poolingWidth(1)
                .channels(channels)
                .outputZeroPoint(u8(outputZeroPoint))
                .testQ8();
            AveragePoolingOperatorTester()
                .batchSize(1)
                .inputHeight(2)
                .inputWidth(poolSize + 2)
                .poolingHeight(1)
                .poolingWidth(poolSize)
                .channels(channels)
                .outputZeroPoint(u8(outputZeroPoint))
                .testQ8();
          }
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_few_channels_with_qmin() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .qmin(128)
              .testQ8();
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 1)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .qmin(128)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_unit_batch_few_channels_with_qmax() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .qmax(128)
              .testQ8();
          AveragePoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 1)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .qmax(128)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_small_batch_many_channels_pool() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(3)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .testQ8();
          AveragePoolingOperatorTester()
              .batchSize(3)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_small_batch_many_channels_pool_with_input_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels += 3) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(3)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .inputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
              .testQ8();
          AveragePoolingOperatorTester()
              .batchSize(3)
              .inputHeight(2)
              .inputWidth(poolSize + 1)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .inputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_small_batch_many_channels_pool_with_output_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels += 3) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.q8avgpool.mr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(3)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .outputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
              .testQ8();
          AveragePoolingOperatorTester()
              .batchSize(3)
              .inputHeight(2)
              .inputWidth(poolSize + 1)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .outputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_small_batch_many_channels_large_pool() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels++) {
        for (usize poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
             pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(3)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .testQ8();
          AveragePoolingOperatorTester()
              .batchSize(3)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_small_batch_many_channels_large_pool_with_input_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels += 5) {
        for (usize poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
             pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(3)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .inputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
              .testQ8();
          AveragePoolingOperatorTester()
              .batchSize(3)
              .inputHeight(2)
              .inputWidth(poolSize + 1)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .inputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_small_batch_many_channels_large_pool_with_output_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8avgpool.kr;
           channels <= 3 * pytorch_qnnp_params.q8avgpool.kr;
           channels += 5) {
        for (usize poolSize = pytorch_qnnp_params.q8avgpool.mr + 1; poolSize <=
             pytorch_qnnp_params.q8avgpool.mr + pytorch_qnnp_params.q8avgpool.qr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(3)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .outputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
              .testQ8();
          AveragePoolingOperatorTester()
              .batchSize(3)
              .inputHeight(2)
              .inputWidth(poolSize + 1)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .outputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_small_batch_few_channels() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
             poolSize++) {
          AveragePoolingOperatorTester()
              .batchSize(3)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .testQ8();
          AveragePoolingOperatorTester()
              .batchSize(3)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_small_batch_few_channels_with_input_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
             poolSize += 3) {
          AveragePoolingOperatorTester()
              .batchSize(3)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .inputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
              .testQ8();
          AveragePoolingOperatorTester()
              .batchSize(3)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .inputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_small_batch_few_channels_with_output_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8avgpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.q8avgpool.kr;
             poolSize += 3) {
          AveragePoolingOperatorTester()
              .batchSize(3)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .outputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
              .testQ8();
          AveragePoolingOperatorTester()
              .batchSize(3)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .outputPixelStride(5 * pytorch_qnnp_params.q8avgpool.kr)
              .testQ8();
        }
      }

    */
}

#[test] fn average_pooling_op_setup_increasing_batch() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      AveragePoolingOperatorTester()
          .batchSize(3)
          .nextBatchSize(5)
          .inputHeight(8)
          .inputWidth(8)
          .poolingHeight(5)
          .poolingWidth(3)
          .channels(24)
          .testSetupQ8();

    */
}

#[test] fn average_pooling_op_setup_decreasing_batch() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      AveragePoolingOperatorTester()
          .batchSize(5)
          .nextBatchSize(3)
          .inputHeight(8)
          .inputWidth(8)
          .poolingHeight(5)
          .poolingWidth(3)
          .channels(24)
          .testSetupQ8();

    */
}

#[test] fn average_pooling_op_setup_changing_height() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      AveragePoolingOperatorTester()
          .batchSize(3)
          .inputHeight(8)
          .inputWidth(8)
          .nextInputHeight(9)
          .poolingHeight(5)
          .poolingWidth(3)
          .channels(24)
          .testSetupQ8();
      AveragePoolingOperatorTester()
          .batchSize(3)
          .inputHeight(8)
          .inputWidth(8)
          .nextInputHeight(7)
          .poolingHeight(5)
          .poolingWidth(3)
          .channels(24)
          .testSetupQ8();

    */
}

#[test] fn average_pooling_op_setup_changing_width() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      AveragePoolingOperatorTester()
          .batchSize(3)
          .inputHeight(8)
          .inputWidth(8)
          .nextInputWidth(9)
          .poolingHeight(5)
          .poolingWidth(3)
          .channels(24)
          .testSetupQ8();
      AveragePoolingOperatorTester()
          .batchSize(3)
          .inputHeight(8)
          .inputWidth(8)
          .nextInputWidth(7)
          .poolingHeight(5)
          .poolingWidth(3)
          .channels(24)
          .testSetupQ8();

    */
}

#[test] fn average_pooling_op_setup_swap_height_and_width() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      AveragePoolingOperatorTester()
          .batchSize(3)
          .inputHeight(9)
          .inputWidth(8)
          .nextInputHeight(8)
          .nextInputWidth(9)
          .poolingHeight(5)
          .poolingWidth(3)
          .channels(24)
          .testSetupQ8();

    */
}

