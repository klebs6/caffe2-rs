crate::ix!();

#[inline] pub fn tensor_inference_for_batch_mat_mul(
    def: &OperatorDef,
    input: &Vec<TensorShape>) -> Vec<TensorShape> 
{
    todo!();
    /*
        ArgumentHelper helper(def);
      bool broadcast = helper.GetSingleArgument<int>("broadcast", 0);
      if (!broadcast) {
        const auto ndim = in[0].dims_size();
        CAFFE_ENFORCE_GE(ndim, 2);
        CAFFE_ENFORCE_GE(in[1].dims_size(), 2);
        int a_dim0;
        int b_dim1;
        if (helper.GetSingleArgument<int>("trans_a", 0)) {
          a_dim0 = in[0].dims(ndim - 1);
        } else {
          a_dim0 = in[0].dims(ndim - 2);
        }

        if (helper.GetSingleArgument<int>("trans_b", 0)) {
          b_dim1 = in[1].dims(ndim - 2);
        } else {
          b_dim1 = in[1].dims(ndim - 1);
        }

        auto output_dims =
            vector<int64_t>{in[0].dims().begin(), in[0].dims().end()};
        output_dims[ndim - 2] = a_dim0;
        output_dims[ndim - 1] = b_dim1;

        return vector<TensorShape>{
            CreateTensorShape(vector<int64_t>{output_dims}, in[0].data_type())};
      } else {
        auto ndims_A = in[0].dims_size();
        auto ndims_B = in[1].dims_size();
        std::vector<int64_t> dims_A(ndims_A), dims_B(ndims_B);
        for (int i = 0; i < ndims_A; ++i) {
          dims_A[i] = in[0].dims(i);
        }
        for (int i = 0; i < ndims_B; ++i) {
          dims_B[i] = in[1].dims(i);
        }
        bool A_broadcasted = false, B_broadcasted = false;
        if (ndims_A == 1) {
          dims_A.insert(dims_A.begin(), 1);
          ndims_A = 2;
          A_broadcasted = true;
        }
        if (ndims_B == 1) {
          dims_B.push_back(1);
          ndims_B = 2;
          B_broadcasted = true;
        }
        size_t M, N;
        if (helper.GetSingleArgument<int>("trans_a", 0)) {
          M = dims_A[ndims_A - 1];
        } else {
          M = dims_A[ndims_A - 2];
        }
        if (helper.GetSingleArgument<int>("trans_b", 0)) {
          N = dims_B[ndims_B - 2];
        } else {
          N = dims_B[ndims_B - 1];
        }

        std::vector<int64_t> new_dims;
        if (ndims_A >= ndims_B) {
          new_dims.assign(dims_A.begin(), dims_A.end() - 2);
        } else {
          new_dims.assign(dims_B.begin(), dims_B.end() - 2);
        }
        if (!A_broadcasted) {
          new_dims.push_back(M);
        }
        if (!B_broadcasted) {
          new_dims.push_back(N);
        }
        if (A_broadcasted && B_broadcasted) {
          new_dims.push_back(1);
        }
        return vector<TensorShape>{
            CreateTensorShape(vector<int64_t>{new_dims}, in[0].data_type())};
      }
    */
}

#[inline] pub fn cost_inference_for_batch_mat_mul(
    def: &OperatorDef, 
    input: &Vec<TensorShape>) -> OpSchemaCost 
{
    todo!();
    /*
        CAFFE_ENFORCE_EQ(in.size(), 2U, "BatchMatMul requires two inputs");

      ArgumentHelper helper(def);
      struct OpSchema::Cost c;
      const auto& A = in[0];
      const auto& B = in[1];
      const TensorShape Y = TensorInferenceForBatchMatMul(def, in)[0];

      uint64_t nElemA = nElemFromDim(A);
      uint64_t nElemB = nElemFromDim(B);
      uint64_t nElemY = nElemFromDim(Y);

      auto ndims_A = A.dims_size();
      size_t K;
      if (helper.GetSingleArgument<int>("trans_a", 0)) {
        K = in[0].dims(ndims_A - 2);
      } else {
        K = in[0].dims(ndims_A - 1);
      }

      c.flops = 2 * nElemY * K;
      c.bytes_read = (nElemA + nElemB) * sizeof(A.data_type());
      c.bytes_written = nElemY * sizeof(Y.data_type());
      c.params_bytes = 0;
      return c;
    */
}
