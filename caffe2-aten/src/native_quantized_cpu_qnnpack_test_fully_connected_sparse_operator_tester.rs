// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/test/fully-connected-sparse-operator-tester.h]

pub fn fill_block_sparse_weights(
        b:              *mut u8,
        N:              usize,
        K:              usize,
        row_block_size: usize,
        col_block_size: usize,
        sparsity:       f32,
        zero_points:    *const u8)  {
    
    todo!();
        /*
            random_device randomDevice;
        auto rng = mt19937(randomDevice());
        bernoulli_distribution dist{sparsity};
        for (u32 n = 0; n < N ; n += row_block_size) {
          for (u32 k = 0; k < K; k += col_block_size) {
            if (dist(rng)) {
              for (u32 nb = 0; (nb < row_block_size) && (n + nb < N); ++nb) {
                for (u32 kb = 0; (kb < col_block_size) && (k + kb < K); ++kb) {
                  *(b + (n + nb) * K + k + kb) = zero_points[n + nb];
                }
              }
            }
          }
        }
        */
}

/**
  | Temp Debug utils that will be removed
  | later
  |
  */
pub fn print_matrix_u8(
        name: *const u8,
        a:    *const u8,
        M:    usize,
        N:    usize)  {
    
    todo!();
        /*
            cout << "Matrix START:" << name << "...\n";
        for (u32 m = 0; m < M ; ++m) {
          for (u32 n = 0; n < N; n++) {
            cout << (const u32)(*(a + m * N + n)) << ", ";
          }
          cout << endl;
        }
        cout << "Matrix END...\n\n";
        */
}

pub fn print_matrix_f32(
    name: *const u8,
    a:    *const f32,
    M:    usize,
    N:    usize)  {
    
    todo!();
        /*
            cout << "Matrix START:" << name << "...\n";
        for (u32 m = 0; m < M ; ++m) {
          for (u32 n = 0; n < N; n++) {
            cout << (*(a + m * N + n)) << ", ";
          }
          cout << endl;
        }
        cout << "Matrix END...\n\n";
        */
}

pub enum Mode {
    Dynamic,
    Runtime,
}

pub struct FullyConnectedSparseOperatorTester {
    input_channels:  usize, // default = { 1 }
    input_stride:    usize, // default = { 0 }
    output_channels: usize, // default = { 1 }
    output_stride:   usize, // default = { 0 }
    batch_size:      usize, // default = { 1 }
    qmin:            u8, // default = { 0 }
    qmax:            u8, // default = { 255 }
    iterations:      usize, // default = { 1 }
    sparsity:        f32, // default = { 0.7f }
    row_block_size:  usize, // default = { 1 }
    col_block_size:  usize, // default = { 4 }
}

impl FullyConnectedSparseOperatorTester {
    
    #[inline] pub fn input_channels(&mut self, input_channels: usize) -> &mut FullyConnectedSparseOperatorTester {
        
        todo!();
        /*
            assert(inputChannels >= 1);
        this->inputChannels_ = inputChannels;
        return *this;
        */
    }
    
    #[inline] pub fn input_channels(&self) -> usize {
        
        todo!();
        /*
            return this->inputChannels_;
        */
    }
    
    #[inline] pub fn output_channels(&mut self, output_channels: usize) -> &mut FullyConnectedSparseOperatorTester {
        
        todo!();
        /*
            assert(outputChannels >= 1);
        this->outputChannels_ = outputChannels;
        return *this;
        */
    }
    
    #[inline] pub fn output_channels(&self) -> usize {
        
        todo!();
        /*
            return this->outputChannels_;
        */
    }
    
    #[inline] pub fn batch_size(&mut self, batch_size: usize) -> &mut FullyConnectedSparseOperatorTester {
        
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
    
    #[inline] pub fn input_stride(&mut self, input_stride: usize) -> &mut FullyConnectedSparseOperatorTester {
        
        todo!();
        /*
            assert(inputStride >= 1);
        this->inputStride_ = inputStride;
        return *this;
        */
    }
    
    #[inline] pub fn input_stride(&self) -> usize {
        
        todo!();
        /*
            if (this->inputStride_ == 0) {
          return inputChannels();
        } else {
          assert(this->inputStride_ >= inputChannels());
          return this->inputStride_;
        }
        */
    }
    
    #[inline] pub fn output_stride(&mut self, output_stride: usize) -> &mut FullyConnectedSparseOperatorTester {
        
        todo!();
        /*
            assert(outputStride >= 1);
        this->outputStride_ = outputStride;
        return *this;
        */
    }
    
    #[inline] pub fn output_stride(&self) -> usize {
        
        todo!();
        /*
            if (this->outputStride_ == 0) {
          return outputChannels();
        } else {
          assert(this->outputStride_ >= outputChannels());
          return this->outputStride_;
        }
        */
    }
    
    #[inline] pub fn qmin(&mut self, qmin: u8) -> &mut FullyConnectedSparseOperatorTester {
        
        todo!();
        /*
            this->qmin_ = qmin;
        return *this;
        */
    }
    
    #[inline] pub fn qmin(&self) -> u8 {
        
        todo!();
        /*
            return this->qmin_;
        */
    }
    
    #[inline] pub fn qmax(&mut self, qmax: u8) -> &mut FullyConnectedSparseOperatorTester {
        
        todo!();
        /*
            this->qmax_ = qmax;
        return *this;
        */
    }
    
    #[inline] pub fn qmax(&self) -> u8 {
        
        todo!();
        /*
            return this->qmax_;
        */
    }
    
    #[inline] pub fn iterations(&mut self, iterations: usize) -> &mut FullyConnectedSparseOperatorTester {
        
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
    
    #[inline] pub fn row_block_size(&mut self, block_size: usize) -> &mut FullyConnectedSparseOperatorTester {
        
        todo!();
        /*
            this->rowBlockSize_ = block_size;
        return *this;
        */
    }
    
    #[inline] pub fn col_block_size(&mut self, block_size: usize) -> &mut FullyConnectedSparseOperatorTester {
        
        todo!();
        /*
            this->colBlockSize_ = block_size;
        return *this;
        */
    }
    
    #[inline] pub fn sparsity(&mut self, s: f32) -> &mut FullyConnectedSparseOperatorTester {
        
        todo!();
        /*
            this->sparsity_ = s;
        return *this;
        */
    }
    
    #[inline] pub fn row_block_size(&self) -> usize {
        
        todo!();
        /*
            return this->rowBlockSize_;
        */
    }
    
    #[inline] pub fn col_block_size(&self) -> usize {
        
        todo!();
        /*
            return this->colBlockSize_;
        */
    }
    
    #[inline] pub fn sparsity(&self) -> f32 {
        
        todo!();
        /*
            return this->sparsity_;
        */
    }
    
    pub fn testq8(&self, mode: Mode)  {
        
        todo!();
        /*
            random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto s32rng =
            bind(uniform_int_distribution<i32>(-10000, 10000), rng);
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);
        auto f32rng =
            bind(uniform_real_distribution<float>(1, 5), rng);

        vector<u8> input(
            (batchSize() - 1) * inputStride() + inputChannels() + 8);
        vector<u8> kernel(outputChannels() * inputChannels());
        vector<i32> bias(outputChannels());
        vector<u8> output(
            (batchSize() - 1) * outputStride() + outputChannels());
        vector<float> output_dynamic(output.size());
        vector<i32> accumulators(batchSize() * outputChannels());
        vector<float> accumulators_float(batchSize() * outputChannels());

        const u8* const inputPtr = input.data();
        const u8 inputZeroPoint = 127;
        // Make number of output channels multiple of 8.
        // This is the least common denominator for SSE/ARM kernels we have.
        usize num_zero_points_padded = outputChannels() + 8;
        vector<u8> kernelZeroPoints(num_zero_points_padded, 127);

        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(input.begin(), input.end(), ref(u8rng));
          generate(bias.begin(), bias.end(), ref(s32rng));
          generate(kernelZeroPoints.begin(), kernelZeroPoints.end(), ref(u8rng));

          u8 max_elem, min_elem;
          do {
            generate(kernel.begin(), kernel.end(), ref(u8rng));
            fillBlockSparseWeights(
                kernel.data(),
                outputChannels(),
                inputChannels(),
                rowBlockSize(),
                colBlockSize(),
                sparsity(),
                kernelZeroPoints.data());
            max_elem = *max_element(kernel.cbegin(), kernel.cend());
            min_elem = *min_element(kernel.cbegin(), kernel.cend());
          } while (max_elem == min_elem);

          unique_ptr<qnnpack_BCSRMatrix> bcsr_matrix =
            qnnpack_generateBlockCSRMatrix(
                kernel.data(),
                outputChannels(),
                inputChannels(),
                rowBlockSize(),
                colBlockSize(),
                kernelZeroPoints.data());

          fill(output.begin(), output.end(), 0xA5);
          fill(output_dynamic.begin(), output_dynamic.end(), 0.0f);
          fill(accumulators.begin(), accumulators.end(), 0);

          for (usize i = 0; i < batchSize(); i++) {
            for (usize oc = 0; oc < outputChannels(); oc++) {
              accumulators[i * outputChannels() + oc] = bias[oc];
            }
          }
          for (usize i = 0; i < batchSize(); i++) {
            for (usize oc = 0; oc < outputChannels(); oc++) {
              for (usize ic = 0; ic < inputChannels(); ic++) {
                accumulators[i * outputChannels() + oc] +=
                    (i32(inputPtr[i * inputStride() + ic]) -
                     i32(inputZeroPoint)) *
                    (i32(kernel[oc * inputChannels() + ic]) -
                     i32(kernelZeroPoints[oc]));
              }
            }
          }

          // Create dummy min/max for empty inputs.
          // These are only used to compute scale and zero point,
          // and real callers will just pull those values from the model.
          const i32 accumulatorsMin = accumulators.empty()
              ? 0
              : *min_element(accumulators.cbegin(), accumulators.cend());
          const i32 accumulatorsMax = accumulators.empty()
              ? 900
              : *max_element(accumulators.cbegin(), accumulators.cend());

          const double outputScale =
              double(u32(accumulatorsMax - accumulatorsMin)) / 255.0;
          const u8 outputZeroPoint = u8(max(
              min(
                  lrint(
                      127.5 -
                      0.5 * double(accumulatorsMin + accumulatorsMax) /
                          outputScale),
                  long(u8::max)),
              long(u8::min)));

          ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
          // 1 bcz input_scale and kernel_scale are both 1.
          vector<float>
            requantization_scales(num_zero_points_padded, 1.0 * 1.0 / outputScale);
          auto scale_generator = [&]() -> float {return (f32rng()/outputScale);};
          generate(
              requantization_scales.begin(),
              requantization_scales.end(),
              ref(scale_generator));

          switch(mode) {
            case Mode::Runtime:
              break;
            case Mode::Dynamic: {
                // Attention! Bias size must be a multiple of 8.
                constexpr usize kBiasSizeMultiple = 8u;
                vector<float, AlignedAllocator<float, 32>> bias_float(
                  (bias.size() + (kBiasSizeMultiple - 1)) & -kBiasSizeMultiple);
                copy(bias.cbegin(), bias.cend(), bias_float.begin());

                pytorch_qnnp_operator_t sparse_gemm = nullptr;

                ASSERT_EQ(
                    pytorch_qnnp_status_success,
                    pytorch_qnnp_create_fully_connected_sparse_dq_nc_q8(
                        inputChannels(),
                        outputChannels(),
                        inputZeroPoint,
                        kernelZeroPoints.data(),
                        bcsr_matrix->col_indices.data(),
                        bcsr_matrix->row_values.data(),
                        bcsr_matrix->values.data(),
                        bcsr_matrix->row_block_size,
                        bcsr_matrix->col_block_size,
                        outputZeroPoint,
                        qmin(),
                        qmax(),
                        0,
                        requantization_scales.data(),
                        false,
                        &sparse_gemm));

                ASSERT_EQ(
                    pytorch_qnnp_status_success,
                    pytorch_qnnp_setup_fully_connected_sparse_dq_nc_q8(
                        sparse_gemm,
                        batchSize(),
                        inputPtr,
                        inputStride(),
                        bias_float.data(),
                        output_dynamic.data(),
                        outputStride()));

                ASSERT_EQ(
                    pytorch_qnnp_status_success,
                    pytorch_qnnp_run_operator(sparse_gemm, nullptr /* thread pool */));

                ASSERT_EQ(
                    pytorch_qnnp_status_success,
                    pytorch_qnnp_delete_operator(sparse_gemm));
                sparse_gemm = nullptr;

                break;
              }
            default:
              // Undefined!
              ASSERT_TRUE(false);
          }

          switch (mode) {
            case Mode::Runtime:
              break;
            case Mode::Dynamic:
            {
              // Bias is added post scaling, as float.
              for (usize i = 0; i < batchSize(); i++) {
                for (usize oc = 0; oc < outputChannels(); oc++) {
                  accumulators[i * outputChannels() + oc] -= bias[oc];
                  accumulators_float[i * outputChannels() + oc] =
                    (float)accumulators[i * outputChannels() + oc] *
                      requantization_scales[oc] + float(bias[oc]);
                }
              }
              for (usize i = 0; i < batchSize(); i++) {
                for (usize c = 0; c < outputChannels(); c++) {
                  ASSERT_EQ(
                      output_dynamic[i * outputChannels() + c],
                      accumulators_float[i * outputChannels() + c])
                      << "at " << i << ", " << c
                      << ": reference = " <<
                      accumulators_float[i * outputChannels() + c]
                      << ", optimized = " << output_dynamic[i * outputChannels() + c];
                }
              }
            }
            break;

            default:
              // Undefined!
              ASSERT_TRUE(false);
          }
        }
        */
    }
    
    pub fn testq8_prepacked(&self, mode: Mode)  {
        
        todo!();
        /*
            random_device randomDevice;
        auto rng = mt19937(randomDevice());
        auto s32rng =
            bind(uniform_int_distribution<i32>(-10000, 10000), rng);
        auto u8rng = bind(uniform_int_distribution<u8>(), rng);
        auto f32rng =
            bind(uniform_real_distribution<float>(1, 5), rng);

        vector<u8> input(
            (batchSize() - 1) * inputStride() + inputChannels() + 8);
        vector<u8> kernel(outputChannels() * inputChannels());
        vector<i32> bias(outputChannels());
        vector<u8> output(
            (batchSize() - 1) * outputStride() + outputChannels());
        vector<float> output_dynamic(output.size());
        vector<i32> accumulators(batchSize() * outputChannels());
        vector<float> accumulators_float(batchSize() * outputChannels());

        const u8* const inputPtr = input.data();
        const u8 inputZeroPoint = 127;
        // Make number of output channels multiple of 8.
        // This is the least common denominator for SSE/ARM kernels we have.
        usize num_zero_points_padded = outputChannels() + 8;
        vector<u8> kernelZeroPoints(num_zero_points_padded, 127);

        for (usize iteration = 0; iteration < iterations(); iteration++) {
          generate(input.begin(), input.end(), ref(u8rng));
          generate(bias.begin(), bias.end(), ref(s32rng));
          generate(kernelZeroPoints.begin(), kernelZeroPoints.end(), ref(u8rng));

          u8 max_elem, min_elem;
          do {
            generate(kernel.begin(), kernel.end(), ref(u8rng));
            fillBlockSparseWeights(
                kernel.data(),
                outputChannels(),
                inputChannels(),
                rowBlockSize(),
                colBlockSize(),
                sparsity(),
                kernelZeroPoints.data());
            max_elem = *max_element(kernel.cbegin(), kernel.cend());
            min_elem = *min_element(kernel.cbegin(), kernel.cend());
          } while (max_elem == min_elem);
          unique_ptr<qnnpack_BCSRMatrix> bcsr_matrix =
            qnnpack_generateBlockCSRMatrix(
                kernel.data(),
                outputChannels(),
                inputChannels(),
                rowBlockSize(),
                colBlockSize(),
                kernelZeroPoints.data());

          fill(output.begin(), output.end(), 0xA5);
          fill(output_dynamic.begin(), output_dynamic.end(), 0.0f);
          fill(accumulators.begin(), accumulators.end(), 0);

          for (usize i = 0; i < batchSize(); i++) {
            for (usize oc = 0; oc < outputChannels(); oc++) {
              accumulators[i * outputChannels() + oc] = bias[oc];
            }
          }
          for (usize i = 0; i < batchSize(); i++) {
            for (usize oc = 0; oc < outputChannels(); oc++) {
              for (usize ic = 0; ic < inputChannels(); ic++) {
                accumulators[i * outputChannels() + oc] +=
                    (i32(inputPtr[i * inputStride() + ic]) -
                     i32(inputZeroPoint)) *
                    (i32(kernel[oc * inputChannels() + ic]) -
                     i32(kernelZeroPoints[oc]));
              }
            }
          }

          // Create dummy min/max for empty inputs.
          // These are only used to compute scale and zero point,
          // and real callers will just pull those values from the model.
          const i32 accumulatorsMin = accumulators.empty()
              ? 0
              : *min_element(accumulators.cbegin(), accumulators.cend());
          const i32 accumulatorsMax = accumulators.empty()
              ? 900
              : *max_element(accumulators.cbegin(), accumulators.cend());

          const double outputScale =
              double(u32(accumulatorsMax - accumulatorsMin)) / 255.0;
          const u8 outputZeroPoint = u8(max(
              min(
                  lrint(
                      127.5 -
                      0.5 * double(accumulatorsMin + accumulatorsMax) /
                          outputScale),
                  long(u8::max)),
              long(u8::min)));

          ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
          // 1 bcz input_scale and kernel_scale are both 1.
          vector<float>
            requantization_scales(num_zero_points_padded, 1.0 * 1.0 / outputScale);
          auto scale_generator = [&]() -> float {return (f32rng()/outputScale);};
          generate(
              requantization_scales.begin(),
              requantization_scales.end(),
              ref(scale_generator));

          switch(mode) {
            case Mode::Runtime:
              break;
            case Mode::Dynamic: {
                // Attention! Bias size must be a multiple of 8.
                constexpr usize kBiasSizeMultiple = 8u;
                vector<float, AlignedAllocator<float, 32>> bias_float(
                  (bias.size() + (kBiasSizeMultiple - 1)) & -kBiasSizeMultiple);
                copy(bias.cbegin(), bias.cend(), bias_float.begin());

                pytorch_qnnp_operator_t sparse_gemm = nullptr;

                ASSERT_EQ(
                    pytorch_qnnp_status_success,
                    pytorch_qnnp_create_fully_connected_sparse_dq_nc_q8(
                        inputChannels(),
                        outputChannels(),
                        inputZeroPoint,
                        kernelZeroPoints.data(),
                        bcsr_matrix->col_indices.data(),
                        bcsr_matrix->row_values.data(),
                        bcsr_matrix->values.data(),
                        bcsr_matrix->row_block_size,
                        bcsr_matrix->col_block_size,
                        outputZeroPoint,
                        qmin(),
                        qmax(),
                        0,
                        requantization_scales.data(),
                        true,
                        &sparse_gemm));

                ASSERT_EQ(
                    pytorch_qnnp_status_success,
                    pytorch_qnnp_setup_fully_connected_sparse_dq_nc_q8(
                        sparse_gemm,
                        batchSize(),
                        inputPtr,
                        inputStride(),
                        bias_float.data(),
                        output_dynamic.data(),
                        outputStride()));

                ASSERT_EQ(
                    pytorch_qnnp_status_success,
                    pytorch_qnnp_run_operator(sparse_gemm, nullptr /* thread pool */));

                ASSERT_EQ(
                    pytorch_qnnp_status_success,
                    pytorch_qnnp_delete_operator(sparse_gemm));
                sparse_gemm = nullptr;

                break;
              }
            default:
              // Undefined!
              ASSERT_TRUE(false);
          }

          switch (mode) {
            case Mode::Runtime:
              break;
            case Mode::Dynamic:
            {
              // Bias is added post scaling, as float.
              for (usize i = 0; i < batchSize(); i++) {
                for (usize oc = 0; oc < outputChannels(); oc++) {
                  accumulators[i * outputChannels() + oc] -= bias[oc];
                  accumulators_float[i * outputChannels() + oc] =
                    (float)accumulators[i * outputChannels() + oc] *
                      requantization_scales[oc] + float(bias[oc]);
                }
              }

              for (usize i = 0; i < batchSize(); i++) {
                for (usize c = 0; c < outputChannels(); c++) {
                  ASSERT_EQ(
                      output_dynamic[i * outputChannels() + c],
                      accumulators_float[i * outputChannels() + c])
                      << "at " << i << ", " << c
                      << ": reference = " <<
                      accumulators_float[i * outputChannels() + c]
                      << ", optimized = " << output_dynamic[i * outputChannels() + c];
                }
              }
            }
            break;

            default:
              // Undefined!
              ASSERT_TRUE(false);
          }
        }
        */
    }
}
