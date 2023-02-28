crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/channel-shuffle-operator-tester.h]
pub struct ChannelShuffleOperatorTester {
    groups:         usize, // default = { 1 }
    group_channels: usize, // default = { 1 }
    batch_size:     usize, // default = { 1 }
    input_stride:   usize, // default = { 0 }
    output_stride:  usize, // default = { 0 }
    iterations:     usize, // default = { 15 }
}

impl ChannelShuffleOperatorTester {

    
    #[inline] pub fn groups(&mut self, groups: usize) -> &mut ChannelShuffleOperatorTester {
        
        todo!();
        /*
            assert(groups != 0);
        this->groups_ = groups;
        return *this;
        */
    }
    
    #[inline] pub fn groups(&self) -> usize {
        
        todo!();
        /*
            return this->groups_;
        */
    }
    
    #[inline] pub fn group_channels(&mut self, group_channels: usize) -> &mut ChannelShuffleOperatorTester {
        
        todo!();
        /*
            assert(groupChannels != 0);
        this->groupChannels_ = groupChannels;
        return *this;
        */
    }
    
    #[inline] pub fn group_channels(&self) -> usize {
        
        todo!();
        /*
            return this->groupChannels_;
        */
    }
    
    #[inline] pub fn channels(&self) -> usize {
        
        todo!();
        /*
            return groups() * groupChannels();
        */
    }
    
    #[inline] pub fn input_stride(&mut self, input_stride: usize) -> &mut ChannelShuffleOperatorTester {
        
        todo!();
        /*
            assert(inputStride != 0);
        this->inputStride_ = inputStride;
        return *this;
        */
    }
    
    #[inline] pub fn input_stride(&self) -> usize {
        
        todo!();
        /*
            if (this->inputStride_ == 0) {
          return channels();
        } else {
          assert(this->inputStride_ >= channels());
          return this->inputStride_;
        }
        */
    }
    
    #[inline] pub fn output_stride(&mut self, output_stride: usize) -> &mut ChannelShuffleOperatorTester {
        
        todo!();
        /*
            assert(outputStride != 0);
        this->outputStride_ = outputStride;
        return *this;
        */
    }
    
    #[inline] pub fn output_stride(&self) -> usize {
        
        todo!();
        /*
            if (this->outputStride_ == 0) {
          return channels();
        } else {
          assert(this->outputStride_ >= channels());
          return this->outputStride_;
        }
        */
    }
    
    #[inline] pub fn batch_size(&mut self, batch_size: usize) -> &mut ChannelShuffleOperatorTester {
        
        todo!();
        /*
            this->batchSize_ = batchSize;
        return *this;
        */
    }
    
    #[inline] pub fn batch_size(&self) -> usize {
        
        todo!();
        /*
            return this->batchSize_;
        */
    }
    
    #[inline] pub fn iterations(&mut self, iterations: usize) -> &mut ChannelShuffleOperatorTester {
        
        todo!();
        /*
            this->iterations_ = iterations;
        return *this;
        */
    }
    
    #[inline] pub fn iterations(&self) -> usize {
        
        todo!();
        /*
            return this->iterations_;
        */
    }
    
    pub fn testx8(&self)  {
        
        todo!();
        /*
            random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);

        vector<u8> input((batchSize() - 1) * inputStride() + channels());
        vector<u8> output(
            (batchSize() - 1) * outputStride() + channels());
        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(input.begin(), input.end(), ref(u8rng));
          fill(output.begin(), output.end(), 0xA5);

          /* Create, setup, run, and destroy Channel Shuffle operator */
          ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
          pytorch_qnnp_operator_t channel_shuffle_op = nullptr;

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_create_channel_shuffle_nc_x8(
                  groups(), groupChannels(), 0, &channel_shuffle_op));
          ASSERT_NE(nullptr, channel_shuffle_op);

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_setup_channel_shuffle_nc_x8(
                  channel_shuffle_op,
                  batchSize(),
                  input.data(),
                  inputStride(),
                  output.data(),
                  outputStride()));

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_run_operator(
                  channel_shuffle_op, nullptr /* thread pool */));

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_delete_operator(channel_shuffle_op));
          channel_shuffle_op = nullptr;

          /* Verify results */
          for (usize i = 0; i < batchSize(); i++) {
            for (usize g = 0; g < groups(); g++) {
              for (usize c = 0; c < groupChannels(); c++) {
                ASSERT_EQ(
                    u32(input[i * inputStride() + g * groupChannels() + c]),
                    u32(output[i * outputStride() + c * groups() + g]));
              }
            }
          }
        }
        */
    }
}

