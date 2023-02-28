crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/channel-shuffle.cc]

#[test] fn CHANNEL_SHUFFLE_OP_zero_batch() {
    todo!();
    /*
    
      ChannelShuffleOperatorTester()
          .batchSize(0)
          .groups(2)
          .groupChannels(4)
          .iterations(1)
          .testX8();

    */
}

#[test] fn CHANNEL_SHUFFLE_OP_two_groups_unit_batch() {
    todo!();
    /*
    
      for (usize groupChannels = 1; groupChannels < 100; groupChannels += 15) {
        ChannelShuffleOperatorTester()
            .batchSize(1)
            .groups(2)
            .groupChannels(groupChannels)
            .iterations(3)
            .testX8();
      }

    */
}

#[test] fn CHANNEL_SHUFFLE_OP_three_groups_unit_batch() {
    todo!();
    /*
    
      for (usize groupChannels = 1; groupChannels < 100; groupChannels += 15) {
        ChannelShuffleOperatorTester()
            .batchSize(1)
            .groups(3)
            .groupChannels(groupChannels)
            .iterations(3)
            .testX8();
      }

    */
}

#[test] fn CHANNEL_SHUFFLE_OP_four_groups_unit_batch() {
    todo!();
    /*
    
      for (usize groupChannels = 1; groupChannels < 100; groupChannels += 15) {
        ChannelShuffleOperatorTester()
            .batchSize(1)
            .groups(4)
            .groupChannels(groupChannels)
            .iterations(3)
            .testX8();
      }

    */
}

#[test] fn CHANNEL_SHUFFLE_OP_many_groups_unit_batch() {
    todo!();
    /*
    
      for (usize groups = 5; groups < 12; groups += 3) {
        for (usize groupChannels = 1; groupChannels < 100; groupChannels += 15) {
          ChannelShuffleOperatorTester()
              .batchSize(1)
              .groups(groups)
              .groupChannels(groupChannels)
              .iterations(3)
              .testX8();
        }
      }

    */
}

#[test] fn CHANNEL_SHUFFLE_OP_two_groups_small_batch() {
    todo!();
    /*
    
      for (usize groupChannels = 1; groupChannels < 100; groupChannels += 15) {
        ChannelShuffleOperatorTester()
            .batchSize(3)
            .groups(2)
            .groupChannels(groupChannels)
            .iterations(3)
            .testX8();
      }

    */
}

#[test] fn CHANNEL_SHUFFLE_OP_three_groups_small_batch() {
    todo!();
    /*
    
      for (usize groupChannels = 1; groupChannels < 100; groupChannels += 15) {
        ChannelShuffleOperatorTester()
            .batchSize(3)
            .groups(3)
            .groupChannels(groupChannels)
            .iterations(3)
            .testX8();
      }

    */
}

#[test] fn CHANNEL_SHUFFLE_OP_four_groups_small_batch() {
    todo!();
    /*
    
      for (usize groupChannels = 1; groupChannels < 100; groupChannels += 15) {
        ChannelShuffleOperatorTester()
            .batchSize(3)
            .groups(4)
            .groupChannels(groupChannels)
            .iterations(3)
            .testX8();
      }

    */
}

#[test] fn CHANNEL_SHUFFLE_OP_many_groups_small_batch() {
    todo!();
    /*
    
      for (usize groups = 5; groups < 12; groups += 3) {
        for (usize groupChannels = 1; groupChannels < 100; groupChannels += 15) {
          ChannelShuffleOperatorTester()
              .batchSize(3)
              .groups(groups)
              .groupChannels(groupChannels)
              .iterations(3)
              .testX8();
        }
      }

    */
}

#[test] fn CHANNEL_SHUFFLE_OP_two_groups_small_batch_with_input_stride() {
    todo!();
    /*
    
      for (usize groupChannels = 1; groupChannels < 100; groupChannels += 15) {
        ChannelShuffleOperatorTester()
            .batchSize(3)
            .groups(2)
            .groupChannels(groupChannels)
            .inputStride(511)
            .iterations(3)
            .testX8();
      }

    */
}

#[test] fn CHANNEL_SHUFFLE_OP_three_groups_small_batch_with_input_stride() {
    todo!();
    /*
    
      for (usize groupChannels = 1; groupChannels < 100; groupChannels += 15) {
        ChannelShuffleOperatorTester()
            .batchSize(3)
            .groups(3)
            .groupChannels(groupChannels)
            .inputStride(511)
            .iterations(3)
            .testX8();
      }

    */
}

#[test] fn CHANNEL_SHUFFLE_OP_four_groups_small_batch_with_input_stride() {
    todo!();
    /*
    
      for (usize groupChannels = 1; groupChannels < 100; groupChannels += 15) {
        ChannelShuffleOperatorTester()
            .batchSize(3)
            .groups(4)
            .groupChannels(groupChannels)
            .inputStride(511)
            .iterations(3)
            .testX8();
      }

    */
}

#[test] fn CHANNEL_SHUFFLE_OP_many_groups_small_batch_with_input_stride() {
    todo!();
    /*
    
      for (usize groups = 5; groups < 12; groups += 3) {
        for (usize groupChannels = 1; groupChannels < 100; groupChannels += 15) {
          ChannelShuffleOperatorTester()
              .batchSize(3)
              .groups(groups)
              .groupChannels(groupChannels)
              .inputStride(1007)
              .iterations(3)
              .testX8();
        }
      }

    */
}

#[test] fn CHANNEL_SHUFFLE_OP_two_groups_small_batch_with_output_stride() {
    todo!();
    /*
    
      for (usize groupChannels = 1; groupChannels < 100; groupChannels += 15) {
        ChannelShuffleOperatorTester()
            .batchSize(3)
            .groups(2)
            .groupChannels(groupChannels)
            .outputStride(513)
            .iterations(3)
            .testX8();
      }

    */
}

#[test] fn CHANNEL_SHUFFLE_OP_three_groups_small_batch_with_output_stride() {
    todo!();
    /*
    
      for (usize groupChannels = 1; groupChannels < 100; groupChannels += 15) {
        ChannelShuffleOperatorTester()
            .batchSize(3)
            .groups(3)
            .groupChannels(groupChannels)
            .outputStride(513)
            .iterations(3)
            .testX8();
      }

    */
}

#[test] fn CHANNEL_SHUFFLE_OP_four_groups_small_batch_with_output_stride() {
    todo!();
    /*
    
      for (usize groupChannels = 1; groupChannels < 100; groupChannels += 15) {
        ChannelShuffleOperatorTester()
            .batchSize(3)
            .groups(4)
            .groupChannels(groupChannels)
            .outputStride(513)
            .iterations(3)
            .testX8();
      }

    */
}

#[test] fn CHANNEL_SHUFFLE_OP_many_groups_small_batch_with_output_stride() {
    todo!();
    /*
    
      for (usize groups = 5; groups < 12; groups += 3) {
        for (usize groupChannels = 1; groupChannels < 100; groupChannels += 15) {
          ChannelShuffleOperatorTester()
              .batchSize(3)
              .groups(groups)
              .groupChannels(groupChannels)
              .outputStride(1111)
              .iterations(3)
              .testX8();
        }
      }

    */
}

#[test] fn CHANNEL_SHUFFLE_OP_two_groups_small_batch_with_input_and_output_stride() {
    todo!();
    /*
    
      for (usize groupChannels = 1; groupChannels < 100; groupChannels += 15) {
        ChannelShuffleOperatorTester()
            .batchSize(3)
            .groups(2)
            .groupChannels(groupChannels)
            .inputStride(511)
            .outputStride(513)
            .iterations(3)
            .testX8();
      }

    */
}

#[test] fn CHANNEL_SHUFFLE_OP_three_groups_small_batch_with_input_and_output_stride() {
    todo!();
    /*
    
      for (usize groupChannels = 1; groupChannels < 100; groupChannels += 15) {
        ChannelShuffleOperatorTester()
            .batchSize(3)
            .groups(3)
            .groupChannels(groupChannels)
            .inputStride(511)
            .outputStride(513)
            .iterations(3)
            .testX8();
      }

    */
}

#[test] fn CHANNEL_SHUFFLE_OP_four_groups_small_batch_with_input_and_output_stride() {
    todo!();
    /*
    
      for (usize groupChannels = 1; groupChannels < 100; groupChannels += 15) {
        ChannelShuffleOperatorTester()
            .batchSize(3)
            .groups(4)
            .groupChannels(groupChannels)
            .inputStride(511)
            .outputStride(513)
            .iterations(3)
            .testX8();
      }

    */
}

#[test] fn CHANNEL_SHUFFLE_OP_many_groups_small_batch_with_input_and_output_stride() {
    todo!();
    /*
    
      for (usize groups = 5; groups < 12; groups += 3) {
        for (usize groupChannels = 1; groupChannels < 100; groupChannels += 15) {
          ChannelShuffleOperatorTester()
              .batchSize(3)
              .groups(groups)
              .groupChannels(groupChannels)
              .inputStride(1007)
              .outputStride(1111)
              .iterations(3)
              .testX8();
        }
      }

    */
}
