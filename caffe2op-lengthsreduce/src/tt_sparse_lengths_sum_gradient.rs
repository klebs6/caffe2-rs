crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct TTSparseLengthsSumGradientOp<T,Context> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

impl<T,Context> TTSparseLengthsSumGradientOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
          const auto& core0 = Input(0);
          const auto& core1 = Input(1);
          const auto& core2 = Input(2);
          const auto& lengths = Input(3);
          const auto& core0_out = Input(4);
          const auto& core1_out = Input(5);
          const auto& index_out = Input(6);
          const auto& dY = Input(7);

          const int* lengths_data = lengths.template data<int>();
          const T* dY_data = dY.template data<T>();

          // restore the arguments from shape
          const int64_t bs = index_out.size(0);
          const int64_t emb_size = dY.size(1);
          const int64_t num_segments = lengths.size(0);

          auto core0_shape = core0.sizes().vec();
          auto core1_shape = core1.sizes().vec();
          auto core2_shape = core2.sizes().vec();
          auto core0_out_shape = core0_out.sizes().vec();
          auto core1_out_shape = core1_out.sizes().vec();

          auto* dCore0 = Output(0, core0_shape, at::dtype<T>());
          auto* dCore1 = Output(1, core1_shape, at::dtype<T>());
          auto* dCore2 = Output(2, core2_shape, at::dtype<T>());

          T* dCore0_data = dCore0->template mutable_data<T>();
          T* dCore1_data = dCore1->template mutable_data<T>();
          T* dCore2_data = dCore2->template mutable_data<T>();

          memset(
              dCore0_data,
              0.0f,
              sizeof(T) *
                  accumulate(
                      core0_shape.begin(), core0_shape.end(), 1, std::multiplies<T>()));
          memset(
              dCore1_data,
              0.0f,
              sizeof(T) *
                  accumulate(
                      core1_shape.begin(), core1_shape.end(), 1, std::multiplies<T>()));
          memset(
              dCore2_data,
              0.0f,
              sizeof(T) *
                  accumulate(
                      core2_shape.begin(), core2_shape.end(), 1, std::multiplies<T>()));

          int64_t* index_out_data = index_out.template mutable_data<int64_t>();

          vector<vector<int64_t>> index_slice(bs, vector<int64_t>(3, 0));
          for (int64_t b = 0; b < bs; b++) {
            memcpy(index_slice[b].data(), index_out_data + b * 3, 3 * sizeof(int64_t));
          }

          vector<const T*> A_ptr(bs);
          vector<T*> B_ptr(bs);
          vector<T*> C_ptr(bs);
          // size of each batch
          int64_t num_of_elements = 0;

          // construct the ranks
          // expand the gradient into all indices
          vector<vector<T>> core2_out_grad(bs, vector<T>(emb_size, 0));
          int64_t data_index = 0;
          for (int64_t range_index = 0; range_index < num_segments; ++range_index) {
            for (int64_t start = data_index;
                 data_index < start + lengths_data[range_index];
                 ++data_index) {
              memcpy(
                  core2_out_grad[data_index].data(),
                  dY_data + range_index * emb_size,
                  emb_size * sizeof(T));
            }
          }

          // =======================================================
          // Calculate dCore2_data:
          // 1) Transpose core1_out and multiply iwth core2_out_grad
          // 2)  add to dCore2_data
          vector<vector<T>> dCore2_data_slice_grad(
              bs, vector<T>(core2_shape[1] * core2_shape[2] * core2_shape[3], 0));
          const T* core1_out_data = core1_out.template data<T>();
          // const T* core1_out_p[bs];
          for (int64_t b = 0; b < bs; b++) {
            A_ptr[b] = core1_out_data + b * core1_out.size(1) * core1_out.size(2);
            B_ptr[b] = core2_out_grad[b].data();
            C_ptr[b] = dCore2_data_slice_grad[b].data();
          }

          math::GemmBatched<T, CPUContext>(
              CblasTrans,
              CblasNoTrans,
              bs,
              core2.size(1), // M
              core2.size(2) * core2.size(3), // N
              core1_out.size(1), // K
              1.0f,
              const_cast<const T**>(A_ptr.data()),
              const_cast<const T**>(B_ptr.data()),
              0.0f,
              C_ptr.data(),
              &context_);

          // update the corresponding slice
          num_of_elements = core2_shape[1] * core2_shape[2] * core2_shape[3];

          T* core2_data = core2.template mutable_data<T>();
          vector<vector<T>> core2_slice(
              bs, vector<T>(core2_shape[1] * core2_shape[2] * core2_shape[3], 0));

          for (int64_t b = 0; b < bs; b++) {
            for (int i = 0; i < num_of_elements; i++) {
              dCore2_data[index_slice[b][2] * num_of_elements + i] += C_ptr[b][i];
            }
            memcpy(
                core2_slice[b].data(),
                core2_data + index_slice[b][2] * num_of_elements,
                sizeof(T) * num_of_elements);
          }

          // Calculate core1_out_grad
          vector<vector<T>> core1_out_grad(
              bs, vector<T>(core1_out_shape[1] * core1_out_shape[2], 0));

          for (int64_t b = 0; b < bs; b++) {
            A_ptr[b] = core2_out_grad[b].data();
            B_ptr[b] = core2_slice[b].data();
            C_ptr[b] = core1_out_grad[b].data();
          }

          math::GemmBatched<T, CPUContext>(
              CblasNoTrans,
              CblasTrans,
              bs,
              core1_out.size(1), // M
              core2_shape[1], // N
              core2_shape[2] * core2_shape[3], // K
              1.0f,
              const_cast<const T**>(A_ptr.data()),
              const_cast<const T**>(B_ptr.data()),
              0.0f,
              C_ptr.data(),
              &context_);

          // =======================================================
          // Calcuate dCore1_data:
          // 1) Transpose core1_out_grad and multiply with core0_out
          // 2) Transpose the result and then add to dCore1_data
          vector<vector<T>> dCore1_data_slice_grad(
              bs, vector<T>(core1_shape[1] * core1_shape[2] * core1_shape[3], 0));
          const T* core0_out_data = core0_out.template data<T>();
          for (int64_t b = 0; b < bs; b++) {
            A_ptr[b] = core0_out_data + b * core0_out.size(1) * core0_out.size(2);
            B_ptr[b] = core1_out_grad[b].data();
            C_ptr[b] = dCore1_data_slice_grad[b].data();
          }

          math::GemmBatched<T, CPUContext>(
              CblasTrans,
              CblasNoTrans,
              bs,
              core1.size(1), // M
              core1.size(2) * core1.size(3), // N
              core0_out.size(1), // K
              1.0f,
              const_cast<const T**>(A_ptr.data()),
              const_cast<const T**>(B_ptr.data()),
              0.0f,
              C_ptr.data(),
              &context_);

          // update the corresponding slice
          num_of_elements = core1_shape[1] * core1_shape[2] * core1_shape[3];
          T* core1_data = core1.template mutable_data<T>();
          vector<vector<T>> core1_slice(
              bs, vector<T>(core1_shape[1] * core1_shape[2] * core1_shape[3], 0));

          for (int64_t b = 0; b < bs; b++) {
            for (int i = 0; i < num_of_elements; i++) {
              dCore1_data[index_slice[b][1] * num_of_elements + i] += C_ptr[b][i];
            }
            memcpy(
                core1_slice[b].data(),
                core1_data + index_slice[b][1] * num_of_elements,
                sizeof(T) * num_of_elements);
          }

          // Calcuate core0_out_grad
          vector<vector<T>> core0_out_grad(
              bs, vector<T>(core0_out_shape[1] * core0_out_shape[2], 0));

          for (int64_t b = 0; b < bs; b++) {
            A_ptr[b] = core1_out_grad[b].data();
            B_ptr[b] = core1_slice[b].data();
            C_ptr[b] = core0_out_grad[b].data();
          }

          math::GemmBatched<T, CPUContext>(
              CblasNoTrans,
              CblasTrans,
              bs,
              core0_out.size(1), // M
              core1_shape[1], // N
              core1_shape[2] * core1_shape[3], // K
              1.0f,
              const_cast<const T**>(A_ptr.data()),
              const_cast<const T**>(B_ptr.data()),
              0.0f,
              C_ptr.data(),
              &context_);

          num_of_elements = core0_shape[1] * core0_shape[2] * core0_shape[3];

          for (int64_t b = 0; b < bs; b++) {
            for (int i = 0; i < num_of_elements; i++) {
              dCore0_data[index_slice[b][0] * num_of_elements + i] += C_ptr[b][i];
            }
          }
          return true;
        */
    }
}

