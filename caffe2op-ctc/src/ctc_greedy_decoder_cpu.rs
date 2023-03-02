crate::ix!();

impl CTCGreedyDecoderOp<CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // [max_time_step, batch_size, num_classes]
      auto& inputs = Input(INPUTS);
      // [batch_size]

      // [total_decoded_output]

      const auto inputs_dims = inputs.sizes();
      int32_t max_time_step = inputs_dims[0];
      int32_t batch_size = inputs_dims[1];
      int32_t num_classes = inputs_dims[2];
      // [batch_size]
      const int* seq_len_data =
          (InputSize() == 2) ? Input(SEQ_LEN).data<int>() : nullptr;

      vector<int> values_cach;
      auto* output_len =
          Output(OUTPUT_LEN, vector<int64_t>{batch_size}, at::dtype<int>());
      int* output_len_data = output_len->template mutable_data<int>();

      for (int32_t i = 0; i < batch_size; ++i) {
        int previous_label = 0, t_dec = 0;
        int32_t seq_len_i = (seq_len_data) ? seq_len_data[i] : max_time_step;
        CAFFE_ENFORCE_LE(seq_len_i, max_time_step);
        for (int32_t t = 0; t < seq_len_i; ++t) {
          auto* prob_data = getTensorDataPtr(inputs, t, i);
          int curr_label =
              std::max_element(prob_data, prob_data + num_classes) - prob_data;
          if (curr_label != 0 &&
              (!merge_repeated_ || (previous_label != curr_label))) {
            t_dec++;
            values_cach.push_back(curr_label);
          }
          previous_label = curr_label;
        }
        output_len_data[i] = t_dec;
      }

      int32_t values_cach_size = values_cach.size();
      auto* values =
          Output(VALUES, vector<int64_t>{values_cach_size}, at::dtype<int>());
      int* values_data = values->mutable_data<int>();
      for (size_t i = 0; i < values_cach.size(); ++i) {
        values_data[i] = values_cach.at(i);
      }
      values_cach.clear();

      return true;
        */
    }
}
