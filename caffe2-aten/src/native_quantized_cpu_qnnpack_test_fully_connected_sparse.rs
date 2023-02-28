// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/fully-connected-sparse.cc]

#[macro_export] macro_rules! sparse_op_test {
    ($ROW_BS:expr, $COL_BS:expr) => {
        /*
        
         
        TEST(FULLY_CONNECTED_SPARSE_OP_##ROW_BS ## x ##COL_BS, 
            integration_test_dynamic_prepacked) { 
          FullyConnectedSparseOperatorTester() 
              .batchSize(4) 
              .inputChannels(4) 
              .outputChannels(4) 
              .iterations(3) 
              .rowBlockSize(ROW_BS) 
              .colBlockSize(COL_BS) 
              .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic); 
        } 
         
        TEST(FULLY_CONNECTED_SPARSE_OP_##ROW_BS ## x ##COL_BS, 
            zero_batch_dynamic_prepacked) { 
          FullyConnectedSparseOperatorTester() 
              .batchSize(0) 
              .inputChannels(2) 
              .outputChannels(2) 
              .iterations(1) 
              .rowBlockSize(ROW_BS) 
              .colBlockSize(COL_BS) 
              .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic); 
        } 
         
        TEST(FULLY_CONNECTED_SPARSE_OP_##ROW_BS ## x ##COL_BS, 
            unit_batch_dynamic_prepacked) { 
          FullyConnectedSparseOperatorTester() 
              .batchSize(1) 
              .inputChannels(23) 
              .outputChannels(19) 
              .iterations(3) 
              .rowBlockSize(ROW_BS) 
              .colBlockSize(COL_BS) 
              .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic); 
        } 
         
        TEST(FULLY_CONNECTED_SPARSE_OP_##ROW_BS ## x ##COL_BS, 
            unit_batch_with_qmin_dynamic_prepacked) { 
          FullyConnectedSparseOperatorTester() 
              .batchSize(1) 
              .inputChannels(23) 
              .outputChannels(19) 
              .qmin(128) 
              .iterations(3) 
              .rowBlockSize(ROW_BS) 
              .colBlockSize(COL_BS) 
              .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic); 
        } 
         
        TEST(FULLY_CONNECTED_SPARSE_OP_##ROW_BS ## x ##COL_BS, 
            unit_batch_with_qmax_dynamic_prepacked) { 
          FullyConnectedSparseOperatorTester() 
              .batchSize(1) 
              .inputChannels(23) 
              .outputChannels(19) 
              .qmax(128) 
              .iterations(3) 
              .rowBlockSize(ROW_BS) 
              .colBlockSize(COL_BS) 
              .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic); 
        } 
         
        TEST(FULLY_CONNECTED_SPARSE_OP_##ROW_BS ## x ##COL_BS, 
            unit_batch_with_input_stride_dynamic_prepacked) { 
          FullyConnectedSparseOperatorTester() 
              .batchSize(1) 
              .inputChannels(23) 
              .inputStride(28) 
              .outputChannels(19) 
              .iterations(3) 
              .rowBlockSize(ROW_BS) 
              .colBlockSize(COL_BS) 
              .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic); 
        } 
         
        TEST(FULLY_CONNECTED_SPARSE_OP_##ROW_BS ## x ##COL_BS, 
            unit_batch_with_output_stride_dynamic_prepacked) { 
          FullyConnectedSparseOperatorTester() 
              .batchSize(1) 
              .inputChannels(23) 
              .outputChannels(19) 
              .outputStride(29) 
              .iterations(3) 
              .rowBlockSize(ROW_BS) 
              .colBlockSize(COL_BS) 
              .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic); 
        } 
         
        TEST(FULLY_CONNECTED_SPARSE_OP_##ROW_BS ## x ##COL_BS, 
            small_batch_dynamic_prepacked) { 
          FullyConnectedSparseOperatorTester() 
              .batchSize(12) 
              .inputChannels(23) 
              .outputChannels(19) 
              .iterations(3) 
              .rowBlockSize(ROW_BS) 
              .colBlockSize(COL_BS) 
              .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic); 
        } 
         
        TEST(FULLY_CONNECTED_SPARSE_OP_##ROW_BS ## x ##COL_BS, 
            small_batch_with_qmin_dynamic_prepacked) { 
          FullyConnectedSparseOperatorTester() 
              .batchSize(12) 
              .inputChannels(23) 
              .outputChannels(19) 
              .qmin(128) 
              .iterations(3) 
              .rowBlockSize(ROW_BS) 
              .colBlockSize(COL_BS) 
              .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic); 
        } 
         
        TEST(FULLY_CONNECTED_SPARSE_OP_##ROW_BS ## x ##COL_BS, 
            small_batch_with_qmax_dynamic_prepacked) { 
          FullyConnectedSparseOperatorTester() 
              .batchSize(13) 
              .inputChannels(23) 
              .outputChannels(19) 
              .qmax(128) 
              .iterations(3) 
              .rowBlockSize(ROW_BS) 
              .colBlockSize(COL_BS) 
              .testQ8_prepacked(FullyConnectedSparseOperatorTester::Mode::Dynamic); 
        }
        */
    }
}

sparse_op_test!{1, 4}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
sparse_op_test!{8, 1}
