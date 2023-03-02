crate::ix!();

/**
 | Input `data` is a N * D matrix. Apply box-cox
 | transform for each column. `lambda1` and `lambda2`
 | is of size D that defines the hyper-parameters for
 | the transform of each column `x` of the input
 | `data`:
 |
 |     ln(x + lambda2), if lambda1 == 0
 |     ((x + lambda2)^lambda1 - 1)/lambda1, if lambda1 != 0
 |
 */
pub struct BatchBoxCoxOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;

    storage: OperatorStorage,
    context: Context,

    nonzeros:        Vec<i32>,
    zeros:           Vec<i32>,
    buffers:         Box<CachedBuffers>,
    min_block_size:  i32,
    }

register_cpu_operator!{
    BatchBoxCox, 
    BatchBoxCoxOp<CPUContext>
}

num_inputs!{BatchBoxCox, 3}

num_outputs!{BatchBoxCox, 1}

inputs!{BatchBoxCox, 
    0 => ("data", "input float or double N * D matrix"),
    1 => ("lambda1", "tensor of size D with the same type as data"),
    2 => ("lambda2", "tensor of size D with the same type as data")
}

outputs!{BatchBoxCox, 
    0 => ("output", "output matrix that applied box-cox transform")
}

identical_type_and_shape_of_input!{BatchBoxCox, 0}

allow_inplace!{BatchBoxCox, vec![(0, 0)]}

gradient_not_implemented_yet!{BatchBoxCox}

input_tags!{
    BatchBoxCoxOp {
        Data,
        Lambda1,
        Lambda2
    }
}

impl<Context> BatchBoxCoxOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            min_block_size_(
                this->template GetSingleArgument<int>("min_block_size", 256))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, double>>::call(this, Input(DATA));
        */
    }
}

impl BatchBoxCoxOp<CPUContext> {

    #[inline] pub fn box_cox_naive<T>(
        &mut self, 
        n:             i64,
        d:             i64,
        data_ptr:      *const T,
        lambda1_ptr:   *const T,
        lambda2_ptr:   *const T,
        k_eps:         T,
        output_ptr:    *mut T) 
    {
        todo!();
        /*
            for (int64_t i = 0; i < N; i++) {
            for (int64_t j = 0; j < D; j++, data_ptr++, output_ptr++) {
              T lambda1_v = lambda1_ptr[j];
              T lambda2_v = lambda2_ptr[j];
              T tmp = std::max(*data_ptr + lambda2_v, k_eps);
              if (lambda1_v == 0) {
                *output_ptr = std::log(tmp);
              } else {
                *output_ptr = (std::pow(tmp, lambda1_v) - 1) / lambda1_v;
              }
            }
          }
        */
    }

    #[inline] pub fn box_cox_nonzero_lambda<T>(
        &mut self, 
        d:         i64,
        data_ptr:  *const T,
        lambda1:   *const T,
        lambda2:   *const T,
        k_eps:     T,
        out:       *mut T) 
    {
        todo!();
        /*
            caffe2::math::Add(D, data_ptr, lambda2, out, &context_);
          for (int64_t j = 0; j < D; j++) {
            out[j] = std::max(out[j], k_eps);
          }
          Pow(D, out, lambda1, out);
          for (int64_t j = 0; j < D; j++) {
            out[j] -= 1.0;
          }
          caffe2::math::Div(D, out, lambda1, out, &context_);
        */
    }

    #[inline] pub fn box_cox_zero_lambda<T>(
        &mut self, 
        d:           i64,
        data_ptr:    *const T,
        lambda2:     *const T,
        k_eps:       T,
        output_ptr:  *mut T) 
    {
        todo!();
        /*
            caffe2::math::Add(D, data_ptr, lambda2, output_ptr, &context_);
          for (int64_t j = 0; j < D; j++) {
            output_ptr[j] = std::max(output_ptr[j], k_eps);
          }
          caffe2::math::Log(D, output_ptr, output_ptr, &context_);
        */
    }

    #[inline] pub fn box_cox_mixed_lambda<T>(
        &mut self, 
        data_ptr:    *const T,
        nonzeros:    &Vec<i32>,
        zeros:       &Vec<i32>,
        lambda1:     *const T,
        lambda2:     *const T,
        lambda2_z:   *const T,
        k_eps:       T,
        buffer:      *mut T,
        output_ptr:  *mut T) 
    {
        todo!();
        /*
            PackV(nonzeros.size(), data_ptr, nonzeros.data(), buffer);
          BoxCoxNonzeroLambda(nonzeros.size(), buffer, lambda1, lambda2, k_eps, buffer);
          UnpackV(nonzeros.size(), buffer, output_ptr, nonzeros.data());

          PackV(zeros.size(), data_ptr, zeros.data(), buffer);
          BoxCoxZeroLambda(zeros.size(), buffer, lambda2_z, k_eps, buffer);
          UnpackV(zeros.size(), buffer, output_ptr, zeros.data());
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            auto& data = Input(DATA);
          auto& lambda1 = Input(LAMBDA1);
          auto& lambda2 = Input(LAMBDA2);
          CAFFE_ENFORCE_GE(data.dim(), 1);
          auto N = data.size(0);
          auto D = data.size_from_dim(1);

          auto* output = Output(0, Input(DATA).sizes(), at::dtype<T>());
          auto* output_ptr = output->template mutable_data<T>();

          if (data.numel() <= 0) {
            return true;
          }

          CAFFE_ENFORCE_EQ(lambda1.numel(), D);
          CAFFE_ENFORCE_EQ(lambda2.numel(), D);

          const auto* data_ptr = data.template data<T>();
          const auto* lambda1_ptr = lambda1.template data<T>();
          const auto* lambda2_ptr = lambda2.template data<T>();

          const T k_eps = static_cast<T>(1e-6);

        #ifdef CAFFE2_USE_MKL
          if (min_block_size_ < 1) {
            BoxCoxNaive(N, D, data_ptr, lambda1_ptr, lambda2_ptr, k_eps, output_ptr);
          } else {
            // Find zero-valued columns, since they get special treatment.
            nonzeros_.clear();
            zeros_.clear();
            nonzeros_.reserve(D);
            zeros_.reserve(D);
            for (int64_t j = 0; j < D; j++) {
              if (lambda1_ptr[j] == 0) {
                zeros_.push_back(j);
              } else {
                nonzeros_.push_back(j);
              }
            }

            // Process K rows at a time for effective vectorization with small rows.
            const int K = std::min(N, (min_block_size_ + D - 1) / D);

            // Avoid copying data if all lambda1 values are zero, or if all are nonzero.
            // In each of the three cases here, when K > 1, first process batches of K
            // rows by replicating the input parameters K times. Then finish row-by-row.
            TypedCachedBuffers<T>& b = GetBuffers<T>();
            if (nonzeros_.size() == D) {
              int64_t i = 0;
              if (K > 1) {
                TileArrayIntoVector(lambda1_ptr, D, K, &b.lambda1_);
                TileArrayIntoVector(lambda2_ptr, D, K, &b.lambda2_);
                DCHECK_EQ(K * D, b.lambda1_.size());
                DCHECK_EQ(K * D, b.lambda2_.size());
                for (; i < N - K + 1; i += K, data_ptr += K * D, output_ptr += K * D) {
                  BoxCoxNonzeroLambda(
                      K * D,
                      data_ptr,
                      b.lambda1_.data(),
                      b.lambda2_.data(),
                      k_eps,
                      output_ptr);
                }
              }
              for (; i < N; i++, data_ptr += D, output_ptr += D) {
                BoxCoxNonzeroLambda(
                    D, data_ptr, lambda1_ptr, lambda2_ptr, k_eps, output_ptr);
              }
            } else if (zeros_.size() == D) {
              int64_t i = 0;
              if (K > 1) {
                TileArrayIntoVector(lambda2_ptr, D, K, &b.lambda2_z_);
                DCHECK_EQ(K * D, b.lambda2_z_.size());
                for (; i < N - K + 1; i += K, data_ptr += K * D, output_ptr += K * D) {
                  BoxCoxZeroLambda(
                      K * D, data_ptr, b.lambda2_z_.data(), k_eps, output_ptr);
                }
              }
              for (; i < N; i++, data_ptr += D, output_ptr += D) {
                BoxCoxZeroLambda(D, data_ptr, lambda2_ptr, k_eps, output_ptr);
              }
            } else { // General case of mixed zero and non-zero lambda1 values.
              int n = nonzeros_.size();
              if (K > 1) {
                TileIndicesInPlace(&nonzeros_, 0, K);
                TileIndicesInPlace(&zeros_, 0, K);
              }

              // Gather parameter values into contiguous memory.
              b.lambda1_.resize(nonzeros_.size());
              b.lambda2_.resize(nonzeros_.size());
              b.lambda2_z_.resize(zeros_.size());
              PackV(nonzeros_.size(), lambda1_ptr, nonzeros_.data(), b.lambda1_.data());
              PackV(nonzeros_.size(), lambda2_ptr, nonzeros_.data(), b.lambda2_.data());
              PackV(zeros_.size(), lambda2_ptr, zeros_.data(), b.lambda2_z_.data());

              int64_t i = 0;
              b.accumulator_.resize(std::max(nonzeros_.size(), zeros_.size()));
              if (K > 1) {
                // Truncate to original size, and re-tile with offsets this time.
                nonzeros_.resize(n);
                zeros_.resize(D - n);
                TileIndicesInPlace(&nonzeros_, D, K);
                TileIndicesInPlace(&zeros_, D, K);
                DCHECK_EQ(nonzeros_.size(), b.lambda1_.size());
                DCHECK_EQ(nonzeros_.size(), b.lambda2_.size());
                DCHECK_EQ(zeros_.size(), b.lambda2_z_.size());
                for (; i < N - K + 1; i += K, data_ptr += K * D, output_ptr += K * D) {
                  BoxCoxMixedLambda(
                      data_ptr,
                      nonzeros_,
                      zeros_,
                      b.lambda1_.data(),
                      b.lambda2_.data(),
                      b.lambda2_z_.data(),
                      k_eps,
                      b.accumulator_.data(),
                      output_ptr);
                }
                // Truncate to original size.
                nonzeros_.resize(n);
                zeros_.resize(D - n);
              }
              for (; i < N; i++, data_ptr += D, output_ptr += D) {
                BoxCoxMixedLambda(
                    data_ptr,
                    nonzeros_,
                    zeros_,
                    b.lambda1_.data(),
                    b.lambda2_.data(),
                    b.lambda2_z_.data(),
                    k_eps,
                    b.accumulator_.data(),
                    output_ptr);
              }
            }
          }
        #else // CAFFE2_USE_MKL
          BoxCoxNaive(N, D, data_ptr, lambda1_ptr, lambda2_ptr, k_eps, output_ptr);
        #endif // CAFFE2_USE_MKL
          return true;
        */
    }
}

