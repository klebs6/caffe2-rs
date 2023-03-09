crate::ix!();

/**
  | The TT-layer serves as a low-rank decomposition
  | of a fully connected layer.
  | 
  | The inputs are the same as to a fully connected
  | layer, but the number of parameters
  | are greatly reduced and forward computation
  | time can be drastically reduced especially
  | for layers with large weight matrices.
  | 
  | The multiplication is computed as a
  | product of the input vector with each
  | of the cores that make up the TT layer.
  | 
  | Given the input sizes (inp_sizes),
  | output sizes(out_sizes), and the ranks
  | of each of the cores (tt_ranks), the
  | ith core will have size:
  | 
  | inp_sizes[i] * tt_ranks[i] * 
  | tt_ranks[i + 1] * out_sizes[i].
  | 
  | The complexity of the computation is
  | dictated by the sizes of inp_sizes,
  | out_sizes, and tt_ranks, where there
  | is the trade off between accuracy of
  | the low-rank decomposition and the
  | speed of the computation.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct TTLinearOp<T,Context,Engine> {

    storage:         OperatorStorage,
    context:         Context,

    bias_multiplier: Tensor,
    inp_sizes:       Vec<i32>,
    out_sizes:       Vec<i32>,
    tt_ranks:        Vec<i32>,
    y_temp:          Box<Blob>,
    phantom:         PhantomData<T>,
    phantomE:        PhantomData<Engine>,
}

num_inputs!{TT, 3}

num_outputs!{TT, 1}

inputs!{TT, 
    0 => ("X", "Input tensor from previous layer with size (M x K), where M is the batch size and K is the input size."),
    1 => ("b", "1D blob containing the bias vector"),
    2 => ("cores", "1D blob containing each individual cores with sizes specified above.")
}

outputs!{TT, 
    0 => ("Y", "Output tensor from previous layer with size (M x N), where M is the batch size and N is the output size.")
}

args!{TT, 
    0 => ("inp_sizes", "(int[]) Input sizes of cores. Indicates the input size of the individual cores; the size of the input vector X must match the product of the inp_sizes array."),
    1 => ("out_sizes", "(int[]) Output sizes of cores. Indicates the output size of the individual cores; the size of the output vector Y must match the product of the out_sizes array."),
    2 => ("tt_ranks", "(int[]) Ranks of cores. Indicates the ranks of the individual cores; lower rank means larger compression, faster computation but reduce accuracy.")
}

impl<T,Context,Engine> TTLinearOp<T,Context,Engine> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            inp_sizes_(this->template GetRepeatedArgument<int>("inp_sizes")),
            out_sizes_(this->template GetRepeatedArgument<int>("out_sizes")),
            tt_ranks_(this->template GetRepeatedArgument<int>("tt_ranks")),
            Y_temp_(unique_ptr<Blob>(new Blob()))
        */
    }

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0); // Input array
        const auto& b = Input(1); // Bias array
        const auto& cores = Input(2); // 1D array containing the TT-cores

        CAFFE_ENFORCE(X.dim() > 1, "Number of dimensions in X: ", X.dim());
        CAFFE_ENFORCE(b.dim() == 1, "Number of dimensions in b: ", b.dim());
        CAFFE_ENFORCE(
            inp_sizes_.size() == out_sizes_.size(),
            "inp_sizes has size: ",
            inp_sizes_.size(),
            ", out_sizes has size: ",
            out_sizes_.size());
        CAFFE_ENFORCE(
            cores.dim() == 1, "Number of dimensions in cores: ", cores.dim());
        // batch size
        const int batch_size = X.dim() > 1 ? X.dim32(0) : 1;

        // dimension d of tensors
        const int d = inp_sizes_.size();

        // Keep track of index of current core in multiplication
        int cores_idx = 0;

        // Temporary buffer to facilitate multiplication of TT-cores with input
        auto Y_buf = BlobGetMutableTensor(Y_temp_.get(), Context::GetDeviceType());
        Y_buf->ResizeLike(X);
        Y_buf->CopyFrom(X);
        Tensor* Y;

        // The overall forward pass involves multiplication with each core, where
        // each core has sizes dictated by inp_sizes_ and out_sizes_. Each core thus
        // has size inp_sizes_[i] * tt_ranks_[i] * tt_ranks_[i + 1] * out_sizes_[i].
        for (int i = (d - 1); i >= 0; --i) {
          int curr_rows = inp_sizes_[i] * tt_ranks_[i + 1];
          int curr_cols = tt_ranks_[i] * out_sizes_[i];

          // TODO Replace by Reshape(), once wrappers are written
          Y_buf->Resize(Y_buf->numel() / curr_rows, curr_rows);
          Y = Output(
              0, {Y_buf->numel() / curr_rows, curr_cols}, at::dtype<float>());

          // Defensive checks
          CAFFE_ENFORCE(Y_buf->numel() % curr_rows == 0, Y_buf->numel(), curr_rows);
          CAFFE_ENFORCE(
              cores_idx + curr_rows * curr_cols <= cores.numel(),
              cores_idx + curr_rows * curr_cols,
              cores.numel());

          // Multiply ith core with the intermediate output
          math::Gemm<float, Context, Engine>(
              CblasNoTrans,
              CblasNoTrans,
              Y_buf->numel() / curr_rows,
              curr_cols,
              curr_rows,
              1,
              Y_buf->template data<float>(),
              cores.template data<float>() + cores_idx,
              0,
              Y->template mutable_data<float>(),
              &context_);

          CAFFE_ENFORCE(Y->numel() % out_sizes_[i] == 0, Y->numel(), out_sizes_[i]);

          // TODO Add GPU support by writing a generic wrapper.
          auto Y_mat = EigenMatrixMap<float>(
              Y->template mutable_data<float>(),
              Y->numel() / out_sizes_[i],
              out_sizes_[i]);
          Y_mat = ConstEigenMatrixMap<float>(
                      Y->template data<float>(),
                      out_sizes_[i],
                      Y->numel() / out_sizes_[i])
                      .transpose()
                      .eval();

          // Resize operation
          Y_buf->Resize(Y->dim32(0), Y->dim32(1));
          context_.template CopyFromCPU<float>(
              Y->numel(),
              Y->template data<float>(),
              Y_buf->template mutable_data<float>());

          cores_idx += curr_rows * curr_cols;
        }

        // TODO Add GPU support by writing a generic wrapper.
        auto Y_mat = EigenMatrixMap<float>(
            Y->template mutable_data<float>(), batch_size, Y->numel() / batch_size);
        Y_mat = ConstEigenMatrixMap<float>(
                    Y->template data<float>(), Y->numel() / batch_size, batch_size)
                    .transpose()
                    .eval();
        // TODO Replace by Reshape(), once wrappers are written
        Y = Output(0, {batch_size, Y->numel() / batch_size}, at::dtype<float>());

        // Check that output size of Y is the element-wise product of out_sizes
        int prod_out_sizes = 1;
        for (int i = 0; i < out_sizes_.size(); i++) {
          prod_out_sizes *= out_sizes_[i];
        }
        CAFFE_ENFORCE(
            Y->dim32(1) == prod_out_sizes,
            "Output dimension of Y: ",
            Y->dim32(1),
            ", product of out_sizes: ",
            prod_out_sizes);

        // Add bias term
        if (bias_multiplier_.numel() != batch_size) {
          // If the helper bias multiplier is not M, reshape and fill it with one.
          ReinitializeTensor(
              &bias_multiplier_,
              {batch_size},
              at::dtype<T>().device(Context::GetDeviceType()));
          math::Set<T, Context>(
              batch_size,
              static_cast<T>(1),
              bias_multiplier_.template mutable_data<T>(),
              &context_);
        }
        math::Gemm<T, Context, Engine>(
            CblasNoTrans,
            CblasNoTrans,
            Y->dim32(0),
            Y->dim32(1),
            1,
            1,
            bias_multiplier_.template data<T>(),
            b.template data<T>(),
            1,
            Y->template mutable_data<T>(),
            &context_);
        return true;
        */
    }
}

