// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/global-average-pooling.cc]

#[test] fn GLOBAL_AVERAGE_POOLING_OP_zero_batch() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      GlobalAveragePoolingOperatorTester()
          .batchSize(0)
          .width(1)
          .channels(8)
          .testQ8();

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_unit_batch_many_channels_small_width() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8gavgpool.nr;
           channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
             width++) {
          GlobalAveragePoolingOperatorTester()
              .batchSize(1)
              .width(width)
              .channels(channels)
              .testQ8();
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_unit_batch_many_channels_small_width_with_input_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8gavgpool.nr;
           channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
             width++) {
          GlobalAveragePoolingOperatorTester()
              .batchSize(1)
              .width(width)
              .channels(channels)
              .inputStride(5 * pytorch_qnnp_params.q8gavgpool.nr)
              .testQ8();
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_unit_batch_many_channels_small_width_with_input_scale() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8gavgpool.nr;
           channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
             width++) {
          for (float inputScale = 0.01f; inputScale < 100.0f;
               inputScale *= 3.14159265f) {
            GlobalAveragePoolingOperatorTester()
                .batchSize(1)
                .width(width)
                .channels(channels)
                .inputScale(inputScale)
                .testQ8();
          }
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_unit_batch_many_channels_small_width_with_input_zero_point() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8gavgpool.nr;
           channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
             width++) {
          for (i32 inputZeroPoint = 0; inputZeroPoint <= 255;
               inputZeroPoint += 51) {
            GlobalAveragePoolingOperatorTester()
                .batchSize(1)
                .width(width)
                .channels(channels)
                .inputZeroPoint(u8(inputZeroPoint))
                .testQ8();
          }
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_unit_batch_many_channels_small_width_with_output_scale() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8gavgpool.nr;
           channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
             width++) {
          for (float outputScale = 0.01f; outputScale < 100.0f;
               outputScale *= 3.14159265f) {
            GlobalAveragePoolingOperatorTester()
                .batchSize(1)
                .width(width)
                .channels(channels)
                .outputScale(outputScale)
                .testQ8();
          }
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_unit_batch_many_channels_small_width_with_output_zero_point() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8gavgpool.nr;
           channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
             width++) {
          for (i32 outputZeroPoint = 0; outputZeroPoint <= 255;
               outputZeroPoint += 51) {
            GlobalAveragePoolingOperatorTester()
                .batchSize(1)
                .width(width)
                .channels(channels)
                .outputZeroPoint(u8(outputZeroPoint))
                .testQ8();
          }
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_unit_batch_many_channels_small_width_with_output_min() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8gavgpool.nr;
           channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
             width++) {
          GlobalAveragePoolingOperatorTester()
              .batchSize(1)
              .width(width)
              .channels(channels)
              .outputMin(128)
              .testQ8();
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_unit_batch_many_channels_small_width_with_output_max() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8gavgpool.nr;
           channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
             width++) {
          GlobalAveragePoolingOperatorTester()
              .batchSize(1)
              .width(width)
              .channels(channels)
              .outputMax(128)
              .testQ8();
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_unit_batch_many_channels_large_width() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8gavgpool.nr;
           channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = pytorch_qnnp_params.q8gavgpool.mr;
             width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
             width++) {
          GlobalAveragePoolingOperatorTester()
              .batchSize(1)
              .width(width)
              .channels(channels)
              .testQ8();
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_unit_batch_many_channels_large_width_with_input_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8gavgpool.nr;
           channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = pytorch_qnnp_params.q8gavgpool.mr;
             width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
             width++) {
          GlobalAveragePoolingOperatorTester()
              .batchSize(1)
              .width(width)
              .channels(channels)
              .inputStride(5 * pytorch_qnnp_params.q8gavgpool.nr)
              .testQ8();
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_unit_batch_many_channels_large_width_with_input_scale() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8gavgpool.nr;
           channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = pytorch_qnnp_params.q8gavgpool.mr;
             width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
             width++) {
          for (float inputScale = 0.01f; inputScale < 100.0f;
               inputScale *= 3.14159265f) {
            GlobalAveragePoolingOperatorTester()
                .batchSize(1)
                .width(width)
                .channels(channels)
                .inputScale(inputScale)
                .testQ8();
          }
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_unit_batch_many_channels_large_width_with_input_zero_point() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8gavgpool.nr;
           channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = pytorch_qnnp_params.q8gavgpool.mr;
             width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
             width++) {
          for (i32 inputZeroPoint = 0; inputZeroPoint <= 255;
               inputZeroPoint += 51) {
            GlobalAveragePoolingOperatorTester()
                .batchSize(1)
                .width(width)
                .channels(channels)
                .inputZeroPoint(u8(inputZeroPoint))
                .testQ8();
          }
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_unit_batch_many_channels_large_width_with_output_scale() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8gavgpool.nr;
           channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = pytorch_qnnp_params.q8gavgpool.mr;
             width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
             width++) {
          for (float outputScale = 0.01f; outputScale < 100.0f;
               outputScale *= 3.14159265f) {
            GlobalAveragePoolingOperatorTester()
                .batchSize(1)
                .width(width)
                .channels(channels)
                .outputScale(outputScale)
                .testQ8();
          }
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_unit_batch_many_channels_large_width_with_output_zero_point() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8gavgpool.nr;
           channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = pytorch_qnnp_params.q8gavgpool.mr;
             width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
             width++) {
          for (i32 outputZeroPoint = 0; outputZeroPoint <= 255;
               outputZeroPoint += 51) {
            GlobalAveragePoolingOperatorTester()
                .batchSize(1)
                .width(width)
                .channels(channels)
                .outputZeroPoint(u8(outputZeroPoint))
                .testQ8();
          }
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_unit_batch_many_channels_large_width_with_output_min() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8gavgpool.nr;
           channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = pytorch_qnnp_params.q8gavgpool.mr;
             width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
             width++) {
          GlobalAveragePoolingOperatorTester()
              .batchSize(1)
              .width(width)
              .channels(channels)
              .outputMin(128)
              .testQ8();
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_unit_batch_many_channels_large_width_with_output_max() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8gavgpool.nr;
           channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = pytorch_qnnp_params.q8gavgpool.mr;
             width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
             width++) {
          GlobalAveragePoolingOperatorTester()
              .batchSize(1)
              .width(width)
              .channels(channels)
              .outputMax(128)
              .testQ8();
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_unit_batch_few_channels() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
             width++) {
          GlobalAveragePoolingOperatorTester()
              .batchSize(1)
              .width(width)
              .channels(channels)
              .testQ8();
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_unit_batch_few_channels_with_input_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
             width++) {
          GlobalAveragePoolingOperatorTester()
              .batchSize(1)
              .width(width)
              .channels(channels)
              .inputStride(5 * pytorch_qnnp_params.q8gavgpool.nr)
              .testQ8();
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_unit_batch_few_channels_with_input_scale() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
             width++) {
          for (float inputScale = 0.01f; inputScale < 100.0f;
               inputScale *= 3.14159265f) {
            GlobalAveragePoolingOperatorTester()
                .batchSize(1)
                .width(width)
                .channels(channels)
                .inputScale(inputScale)
                .testQ8();
          }
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_unit_batch_few_channels_with_input_zero_point() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
             width++) {
          for (i32 inputZeroPoint = 0; inputZeroPoint <= 255;
               inputZeroPoint += 51) {
            GlobalAveragePoolingOperatorTester()
                .batchSize(1)
                .width(width)
                .channels(channels)
                .inputZeroPoint(u8(inputZeroPoint))
                .testQ8();
          }
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_unit_batch_few_channels_with_output_scale() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
             width++) {
          for (float outputScale = 0.01f; outputScale < 100.0f;
               outputScale *= 3.14159265f) {
            GlobalAveragePoolingOperatorTester()
                .batchSize(1)
                .width(width)
                .channels(channels)
                .outputScale(outputScale)
                .testQ8();
          }
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_unit_batch_few_channels_with_output_zero_point() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
             width++) {
          for (i32 outputZeroPoint = 0; outputZeroPoint <= 255;
               outputZeroPoint += 51) {
            GlobalAveragePoolingOperatorTester()
                .batchSize(1)
                .width(width)
                .channels(channels)
                .outputZeroPoint(u8(outputZeroPoint))
                .testQ8();
          }
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_unit_batch_few_channels_with_output_min() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
             width++) {
          GlobalAveragePoolingOperatorTester()
              .batchSize(1)
              .width(width)
              .channels(channels)
              .outputMin(128)
              .testQ8();
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_unit_batch_few_channels_with_output_max() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
             width++) {
          GlobalAveragePoolingOperatorTester()
              .batchSize(1)
              .width(width)
              .channels(channels)
              .outputMax(128)
              .testQ8();
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_small_batch_many_channels_width() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8gavgpool.nr;
           channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
             width++) {
          GlobalAveragePoolingOperatorTester()
              .batchSize(3)
              .width(width)
              .channels(channels)
              .testQ8();
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_small_batch_many_channels_width_with_input_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8gavgpool.nr;
           channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
             width++) {
          GlobalAveragePoolingOperatorTester()
              .batchSize(3)
              .width(width)
              .channels(channels)
              .inputStride(5 * pytorch_qnnp_params.q8gavgpool.nr)
              .testQ8();
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_small_batch_many_channels_width_with_output_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8gavgpool.nr;
           channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = 1; width <= pytorch_qnnp_params.q8gavgpool.mr;
             width++) {
          GlobalAveragePoolingOperatorTester()
              .batchSize(3)
              .width(width)
              .channels(channels)
              .outputStride(5 * pytorch_qnnp_params.q8gavgpool.nr)
              .testQ8();
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_small_batch_many_channels_large_width() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8gavgpool.nr;
           channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = pytorch_qnnp_params.q8gavgpool.mr;
             width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
             width++) {
          GlobalAveragePoolingOperatorTester()
              .batchSize(3)
              .width(width)
              .channels(channels)
              .testQ8();
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_small_batch_many_channels_large_width_with_input_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8gavgpool.nr;
           channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = pytorch_qnnp_params.q8gavgpool.mr;
             width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
             width++) {
          GlobalAveragePoolingOperatorTester()
              .batchSize(3)
              .width(width)
              .channels(channels)
              .inputStride(5 * pytorch_qnnp_params.q8gavgpool.nr)
              .testQ8();
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_small_batch_many_channels_large_width_with_output_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = pytorch_qnnp_params.q8gavgpool.nr;
           channels <= 3 * pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = pytorch_qnnp_params.q8gavgpool.mr;
             width <= 4 * pytorch_qnnp_params.q8gavgpool.mr;
             width++) {
          GlobalAveragePoolingOperatorTester()
              .batchSize(3)
              .width(width)
              .channels(channels)
              .outputStride(5 * pytorch_qnnp_params.q8gavgpool.nr)
              .testQ8();
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_small_batch_few_channels() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
             width++) {
          GlobalAveragePoolingOperatorTester()
              .batchSize(3)
              .width(width)
              .channels(channels)
              .testQ8();
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_small_batch_few_channels_with_input_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
             width++) {
          GlobalAveragePoolingOperatorTester()
              .batchSize(3)
              .width(width)
              .channels(channels)
              .inputStride(5 * pytorch_qnnp_params.q8gavgpool.nr)
              .testQ8();
        }
      }

    */
}


#[test] fn GLOBAL_AVERAGE_POOLING_OP_small_batch_few_channels_with_output_stride() {
    todo!();
    /*
    
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      for (usize channels = 1; channels < pytorch_qnnp_params.q8gavgpool.nr;
           channels++) {
        for (usize width = 1; width <= 2 * pytorch_qnnp_params.q8gavgpool.nr;
             width++) {
          GlobalAveragePoolingOperatorTester()
              .batchSize(3)
              .width(width)
              .channels(channels)
              .outputStride(5 * pytorch_qnnp_params.q8gavgpool.nr)
              .testQ8();
        }
      }

    */
}


