crate::ix!();

impl BatchDenseToSparseOp<f32, CPUContext> {

    #[inline] pub fn fill_in_sparse_values(
        &mut self, 
        batch_size:         i64,
        indice_lengths:     i64,
        lengths_data:       *const i64,
        indices_data:       *const i64,
        dense_data:         *const f32,
        output_data:        *mut f32,
        context:            *mut CPUContext)  
    {
        todo!();
        /*
            int64_t lengths_sum = 0;
      math::Sum<int64_t, CPUContext>(
          batch_size, lengths_data, &lengths_sum, &context_);
      CAFFE_ENFORCE_EQ(lengths_sum, indice_lengths);

      int64_t k = 0;
      for (int64_t i = 0; i < batch_size; ++i) {
        for (int64_t j = 0; j < lengths_data[i]; ++j) {
          CAFFE_ENFORCE(
              indices_data[k] < dense_last_dim_,
              "An indice (",
              indices_data[k],
              ") is larger then last dim of dense (",
              dense_last_dim_,
              ").");
          output_data[k] = dense_data[i * dense_last_dim_ + indices_data[k]];
          k += 1;
        }
      }
        */
    }
}
