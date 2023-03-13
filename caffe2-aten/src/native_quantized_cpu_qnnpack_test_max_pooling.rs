// # vim: ft=none
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/max-pooling.cc]

#[test] fn max_pooling_op_zero_batch() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      MaxPoolingOperatorTester()
          .batchSize(0)
          .inputHeight(2)
          .inputWidth(6)
          .poolingHeight(1)
          .poolingWidth(8)
          .channels(8)
          .testU8();

    */
}

#[test] fn max_pooling_op_unit_batch_many_channels_small_1xm_pool() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_many_channels_small_1xm_pool_with_padding() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels += 3) {
        for (usize poolSize = 3; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
             poolSize++) {
          for (usize paddingLeft = 0; paddingLeft <= 1; paddingLeft++) {
            for (usize paddingRight = 0; paddingRight <= 1; paddingRight++) {
              MaxPoolingOperatorTester()
                  .batchSize(1)
                  .inputHeight(2)
                  .inputWidth(poolSize + 2)
                  .paddingLeft(paddingLeft)
                  .paddingRight(paddingRight)
                  .poolingHeight(1)
                  .poolingWidth(poolSize)
                  .channels(channels)
                  .testU8();
            }
          }
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_many_channels_small_1xm_pool_with_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels += 3) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 4)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .strideWidth(2)
              .channels(channels)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_many_channels_small_1xm_pool_with_dilation() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels += 3) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(2 * poolSize + 1)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .dilationWidth(2)
              .channels(channels)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_many_channels_small_mx1_pool() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_many_channels_small_mx1_pool_with_padding() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels += 3) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
             poolSize++) {
          for (usize paddingTop = 0; paddingTop <= 1; paddingTop++) {
            for (usize paddingBottom = 0; paddingBottom <= 1; paddingBottom++) {
              MaxPoolingOperatorTester()
                  .batchSize(1)
                  .inputHeight(poolSize + 1)
                  .inputWidth(3)
                  .paddingTop(paddingTop)
                  .paddingBottom(paddingBottom)
                  .poolingHeight(poolSize)
                  .poolingWidth(1)
                  .channels(channels)
                  .testU8();
            }
          }
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_many_channels_small_mx1_pool_with_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels += 3) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 3)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .strideHeight(2)
              .channels(channels)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_many_channels_small_mx1_pool_with_dilation() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels += 3) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2 * poolSize)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .dilationHeight(2)
              .channels(channels)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_many_channels_small_pool_with_input_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels += 3) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .inputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
              .testU8();
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .inputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_many_channels_small_pool_with_output_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels += 3) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .outputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
              .testU8();
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .outputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_many_channels_small_pool_with_qmin() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels += 3) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .qmin(192)
              .testU8();
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .qmin(192)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_many_channels_small_pool_with_qmax() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels += 3) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .qmax(192)
              .testU8();
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .qmax(192)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_many_channels_large_1xm_pool() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels++) {
        for (usize poolSize = pytorch_qnnp_params.u8maxpool.mr; poolSize <=
             pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_many_channels_large_1xm_pool_with_padding() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels += 3) {
        for (usize poolSize = 3; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
             poolSize++) {
          for (usize paddingLeft = 0; paddingLeft <= 1; paddingLeft++) {
            for (usize paddingRight = 0; paddingRight <= 1; paddingRight++) {
              MaxPoolingOperatorTester()
                  .batchSize(1)
                  .inputHeight(2)
                  .inputWidth(poolSize + 2)
                  .paddingLeft(paddingLeft)
                  .paddingRight(paddingRight)
                  .poolingHeight(1)
                  .poolingWidth(poolSize)
                  .channels(channels)
                  .testU8();
            }
          }
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_many_channels_large_1xm_pool_with_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels += 3) {
        for (usize poolSize = pytorch_qnnp_params.u8maxpool.mr; poolSize <=
             pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 4)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .strideWidth(2)
              .channels(channels)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_many_channels_large_1xm_pool_with_dilation() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels += 3) {
        for (usize poolSize = pytorch_qnnp_params.u8maxpool.mr; poolSize <=
             pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(2 * poolSize + 1)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .dilationWidth(2)
              .channels(channels)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_many_channels_large_mx1_pool() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels++) {
        for (usize poolSize = pytorch_qnnp_params.u8maxpool.mr; poolSize <=
             pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_many_channels_large_mx1_pool_with_padding() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels += 3) {
        for (usize poolSize = pytorch_qnnp_params.u8maxpool.mr; poolSize <=
             pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
             poolSize++) {
          for (usize paddingTop = 0; paddingTop <= 1; paddingTop++) {
            for (usize paddingBottom = 0; paddingBottom <= 1; paddingBottom++) {
              MaxPoolingOperatorTester()
                  .batchSize(1)
                  .inputHeight(poolSize + 1)
                  .inputWidth(3)
                  .paddingTop(paddingTop)
                  .paddingBottom(paddingBottom)
                  .poolingHeight(poolSize)
                  .poolingWidth(1)
                  .channels(channels)
                  .testU8();
            }
          }
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_many_channels_large_mx1_pool_with_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels += 3) {
        for (usize poolSize = pytorch_qnnp_params.u8maxpool.mr; poolSize <=
             pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 3)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .strideHeight(2)
              .channels(channels)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_many_channels_large_mx1_pool_with_dilation() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels += 3) {
        for (usize poolSize = pytorch_qnnp_params.u8maxpool.mr; poolSize <=
             pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2 * poolSize)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .dilationHeight(2)
              .channels(channels)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_many_channels_large_pool_with_input_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels += 3) {
        for (usize poolSize = pytorch_qnnp_params.u8maxpool.mr; poolSize <=
             pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .inputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
              .testU8();
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .inputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_many_channels_large_pool_with_output_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels += 3) {
        for (usize poolSize = pytorch_qnnp_params.u8maxpool.mr; poolSize <=
             pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .outputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
              .testU8();
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .outputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_many_channels_large_pool_with_qmin() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels += 3) {
        for (usize poolSize = pytorch_qnnp_params.u8maxpool.mr; poolSize <=
             pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .qmin(192)
              .testU8();
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .qmin(192)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_many_channels_large_pool_with_qmax() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels += 3) {
        for (usize poolSize = pytorch_qnnp_params.u8maxpool.mr; poolSize <=
             pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .qmax(192)
              .testU8();
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .qmax(192)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_few_channels_1xm_pool() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_few_channels_1xm_pool_with_padding() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
           channels++) {
        for (usize poolSize = 3; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
             poolSize++) {
          for (usize paddingLeft = 0; paddingLeft <= 1; paddingLeft++) {
            for (usize paddingRight = 0; paddingRight <= 1; paddingRight++) {
              MaxPoolingOperatorTester()
                  .batchSize(1)
                  .inputHeight(2)
                  .inputWidth(poolSize + 2)
                  .paddingLeft(paddingLeft)
                  .paddingRight(paddingRight)
                  .poolingHeight(1)
                  .poolingWidth(poolSize)
                  .channels(channels)
                  .testU8();
            }
          }
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_few_channels_1xm_pool_with_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 4)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .strideWidth(2)
              .channels(channels)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_few_channels_1xm_pool_with_dilation() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(2 * poolSize + 1)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .dilationWidth(2)
              .channels(channels)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_few_channels_mx1_pool() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_few_channels_mx1_pool_with_padding() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
             poolSize++) {
          for (usize paddingTop = 0; paddingTop <= 1; paddingTop++) {
            for (usize paddingBottom = 0; paddingBottom <= 1; paddingBottom++) {
              MaxPoolingOperatorTester()
                  .batchSize(1)
                  .inputHeight(poolSize + 1)
                  .inputWidth(3)
                  .paddingTop(paddingTop)
                  .paddingBottom(paddingBottom)
                  .poolingHeight(poolSize)
                  .poolingWidth(1)
                  .channels(channels)
                  .testU8();
            }
          }
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_few_channels_mx1_pool_with_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 3)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .strideHeight(2)
              .channels(channels)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_few_channels_mx1_pool_with_dilation() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2 * poolSize)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .dilationHeight(2)
              .channels(channels)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_few_channels_with_input_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .inputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
              .testU8();
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .inputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_few_channels_with_output_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .outputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
              .testU8();
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .outputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_few_channels_with_qmin() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .qmin(192)
              .testU8();
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .qmin(192)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_unit_batch_few_channels_with_qmax() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .qmax(192)
              .testU8();
          MaxPoolingOperatorTester()
              .batchSize(1)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .qmax(192)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_small_batch_many_channels_pool() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(3)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .testU8();
          MaxPoolingOperatorTester()
              .batchSize(3)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_small_batch_many_channels_pool_with_input_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels += 3) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(3)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .inputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
              .testU8();
          MaxPoolingOperatorTester()
              .batchSize(3)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .inputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_small_batch_many_channels_pool_with_output_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels += 3) {
        for (usize poolSize = 2; poolSize <= pytorch_qnnp_params.u8maxpool.mr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(3)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .outputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
              .testU8();
          MaxPoolingOperatorTester()
              .batchSize(3)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .outputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_small_batch_many_channels_large_pool() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels++) {
        for (usize poolSize = pytorch_qnnp_params.u8maxpool.mr + 1; poolSize <=
             pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(3)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .testU8();
          MaxPoolingOperatorTester()
              .batchSize(3)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_small_batch_many_channels_large_pool_with_input_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels += 5) {
        for (usize poolSize = pytorch_qnnp_params.u8maxpool.mr + 1; poolSize <=
             pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(3)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .inputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
              .testU8();
          MaxPoolingOperatorTester()
              .batchSize(3)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .inputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_small_batch_many_channels_large_pool_with_output_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.u8maxpool.kr;
           channels <= 3 * pytorch_qnnp_params.u8maxpool.kr;
           channels += 5) {
        for (usize poolSize = pytorch_qnnp_params.u8maxpool.mr + 1; poolSize <=
             pytorch_qnnp_params.u8maxpool.mr + pytorch_qnnp_params.u8maxpool.qr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(3)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .outputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
              .testU8();
          MaxPoolingOperatorTester()
              .batchSize(3)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .outputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_small_batch_few_channels() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
             poolSize++) {
          MaxPoolingOperatorTester()
              .batchSize(3)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .testU8();
          MaxPoolingOperatorTester()
              .batchSize(3)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_small_batch_few_channels_with_input_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
             poolSize += 3) {
          MaxPoolingOperatorTester()
              .batchSize(3)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .inputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
              .testU8();
          MaxPoolingOperatorTester()
              .batchSize(3)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .inputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_small_batch_few_channels_with_output_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.u8maxpool.kr;
           channels++) {
        for (usize poolSize = 2; poolSize <= 2 * pytorch_qnnp_params.u8maxpool.kr;
             poolSize += 3) {
          MaxPoolingOperatorTester()
              .batchSize(3)
              .inputHeight(poolSize + 1)
              .inputWidth(3)
              .poolingHeight(poolSize)
              .poolingWidth(1)
              .channels(channels)
              .outputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
              .testU8();
          MaxPoolingOperatorTester()
              .batchSize(3)
              .inputHeight(2)
              .inputWidth(poolSize + 2)
              .poolingHeight(1)
              .poolingWidth(poolSize)
              .channels(channels)
              .outputPixelStride(5 * pytorch_qnnp_params.u8maxpool.kr)
              .testU8();
        }
      }

    */
}

#[test] fn max_pooling_op_setup_increasing_batch() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      MaxPoolingOperatorTester()
          .batchSize(3)
          .nextBatchSize(5)
          .inputHeight(8)
          .inputWidth(8)
          .poolingHeight(5)
          .poolingWidth(3)
          .channels(24)
          .testSetupU8();

    */
}

#[test] fn max_pooling_op_setup_decreasing_batch() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      MaxPoolingOperatorTester()
          .batchSize(5)
          .nextBatchSize(3)
          .inputHeight(8)
          .inputWidth(8)
          .poolingHeight(5)
          .poolingWidth(3)
          .channels(24)
          .testSetupU8();

    */
}

#[test] fn max_pooling_op_setup_changing_height() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      MaxPoolingOperatorTester()
          .batchSize(3)
          .inputHeight(8)
          .inputWidth(8)
          .nextInputHeight(9)
          .poolingHeight(5)
          .poolingWidth(3)
          .channels(24)
          .testSetupU8();
      MaxPoolingOperatorTester()
          .batchSize(3)
          .inputHeight(8)
          .inputWidth(8)
          .nextInputHeight(7)
          .poolingHeight(5)
          .poolingWidth(3)
          .channels(24)
          .testSetupU8();

    */
}

#[test] fn max_pooling_op_setup_changing_width() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      MaxPoolingOperatorTester()
          .batchSize(3)
          .inputHeight(8)
          .inputWidth(8)
          .nextInputWidth(9)
          .poolingHeight(5)
          .poolingWidth(3)
          .channels(24)
          .testSetupU8();
      MaxPoolingOperatorTester()
          .batchSize(3)
          .inputHeight(8)
          .inputWidth(8)
          .nextInputWidth(7)
          .poolingHeight(5)
          .poolingWidth(3)
          .channels(24)
          .testSetupU8();

    */
}

#[test] fn max_pooling_op_setup_swap_height_and_width() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      MaxPoolingOperatorTester()
          .batchSize(3)
          .inputHeight(9)
          .inputWidth(8)
          .nextInputHeight(8)
          .nextInputWidth(9)
          .poolingHeight(5)
          .poolingWidth(3)
          .channels(24)
          .testSetupU8();

    */
}


