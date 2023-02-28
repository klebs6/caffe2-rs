crate::ix!();

use crate::{
    Tensor,
    CPUContext,
};

/**
  | Gathers every two elements of a vector
  | in a vector of pairs
  |
  */
#[inline] pub fn pair_vec<T>(vals: &Vec<T>) -> Vec<(T,T)> {
    todo!();
    /*
        CAFFE_ENFORCE_EQ(
          vals.size() % 2,
          0,
          "Odd number of params or hiddens given to a bidirectional RNN");
      std::vector<std::pair<T, T>> result;
      result.reserve(vals.size() / 2);
      for (int64_t i = 0; i < vals.size(); i += 2) {
        result.emplace_back(copy_ctor(vals[i]), copy_ctor(vals[i + 1]));
      }
      return result;
    */
}

/// Flattens a vector of pairs
#[inline] pub fn unpair_vec<T>(vals: Vec<(T,T)>) -> Vec<T> {
    todo!();
    /*
        std::vector<T> result;
      result.reserve(vals.size() * 2);
      for (int64_t i = 0; i < vals.size(); i++) {
        result.push_back(std::move(vals[i].first));
        result.push_back(std::move(vals[i].second));
      }
      return result;
    */
}



#[inline] pub fn matmul(
    x: &Tensor,
    w: &Tensor,
    context: *mut CPUContext) -> Tensor 
{
    todo!();
    /*
        const auto canonical_axis = X.canonical_axis_index(1);
      const auto M = X.size_to_dim(canonical_axis);
      const auto K = X.size_from_dim(canonical_axis);
      const auto canonical_axis_w = W.canonical_axis_index(1);
      const int N = W.size_to_dim(canonical_axis_w);
      auto output_size = X.sizes().vec();
      output_size.resize(canonical_axis + 1);
      output_size[canonical_axis] = N;
      Tensor C(output_size, CPU);
      math::Gemm<float, CPUContext>(
          CblasNoTrans,
          CblasTrans,
          M,
          N,
          K,
          1,
          X.template data<float>(),
          W.template data<float>(),
          0,
          C.template mutable_data<float>(),
          context);
      return C;
    */
}


#[inline] pub fn linear(
    x: &Tensor,
    w: &Tensor,
    b: &Tensor,
    context: *mut CPUContext) -> Tensor 
{
    todo!();
    /*
        auto output = matmul(X, W, context);
      if (B) {
        const auto canonical_axis = X.canonical_axis_index(1);
        const auto M = X.size_to_dim(canonical_axis);
        const auto canonical_axis_w = W.canonical_axis_index(1);
        const int N = W.size_to_dim(canonical_axis_w);
        auto bias_multiplier_ = caffe2::empty({M}, CPU);
        math::Set<float, CPUContext>(
            M, 1, bias_multiplier_.template mutable_data<float>(), context);
        math::Gemm<float, CPUContext>(
            CblasNoTrans,
            CblasNoTrans,
            M,
            N,
            1,
            1,
            bias_multiplier_.template data<float>(),
            B.template data<float>(),
            1,
            output.template mutable_data<float>(),
            context);
      }
      return output;
    */
}


#[inline] pub fn chunk(
    input:   &Tensor,
    chunks:  i32,
    axis:    i32,
    context: *mut CPUContext) -> Vec<Tensor> 
{
    todo!();
    /*
        int canonical_axis = input.canonical_axis_index(axis);
      CAFFE_ENFORCE_LT(
          canonical_axis, input.dim(), "Axis not in input ndim range.");
      const int input_channels = input.dim32(canonical_axis);
      CAFFE_ENFORCE_EQ(
          input_channels % chunks,
          0,
          "input channels should be divisible by the number of chunks.");
      auto split_size = input_channels / chunks;
      vector<int64_t> output_dims(input.sizes().vec());
      int before = 1, after = 1;
      for (int i = 0; i < canonical_axis; ++i) {
        before *= input.dim32(i);
      }
      for (int i = canonical_axis + 1; i < input.dim(); ++i) {
        after *= input.dim32(i);
      }
      size_t input_offset = 0;
      std::vector<Tensor> outputs;
      for (int i = 0; i < chunks; ++i) {
        auto axis_dim = split_size;
        output_dims[canonical_axis] = split_size;
        Tensor output(output_dims, CPU);
        math::CopyMatrix<CPUContext>(
            input.itemsize(),
            before,
            axis_dim * after,
            static_cast<const char*>(input.raw_data()) + input_offset,
            input.dim32(canonical_axis) * after,
            output.raw_mutable_data(input.dtype()),
            axis_dim * after,
            context,
            input.dtype().copy());
        input_offset += axis_dim * after * input.itemsize();
        outputs.push_back(std::move(output));
      }
      return outputs;
    */
}

#[inline] pub fn unbind(
    input:   &Tensor,
    axis:    i32,
    context: *mut CPUContext) -> Vec<Tensor> 
{
    
    todo!();
    /*
        // 1 - Chunk the input tensor along the given axis into N chunks where
      // N is the dim(axis)
      auto chunks = chunk(input, input.sizes()[axis], axis, context);
      // 2 - Compute new dimensions
      std::vector<int64_t> newDims = input.sizes().vec();
      newDims.erase(newDims.begin() + axis);

      // 3 - Reshape chunks to drop the extra dimension
      for (int i = 0; i < chunks.size(); i++) {
        CAFFE_ENFORCE_EQ(
            chunks[i].sizes()[axis], 1, "Got an unexpected chunk size");
        chunks[i].Reshape(newDims);
      }
      return chunks;
    */
}

#[inline] pub fn cat(
    tensor_list: &Vec<Tensor>,
    axis:        i32,
    context:     *mut CPUContext) -> Tensor 
{
    todo!();
    /*
        // Adopted from C2's concat operator
      auto input_zero = copy_ctor(tensorList.at(0));
      vector<int64_t> outputDims(input_zero.sizes().vec());
      CAFFE_ENFORCE(outputDims.size() > 0);
      for (int i = 1; i < tensorList.size(); i++) {
        CAFFE_ENFORCE(input_zero.dtype() == tensorList.at(i).dtype());
        outputDims[axis] += tensorList.at(i).sizes()[axis];
      }
      auto output_channels = outputDims[axis];
      Tensor output(outputDims, CPU);
      int before = 1, after = 1;
      for (int i = 0; i < tensorList.at(0).dim(); ++i) {
        if (i == axis) {
          continue;
        }
        int dim = input_zero.dim32(i);
        if (i < axis) {
          before *= dim;
        } else {
          after *= dim;
        }
      }
      size_t output_offset = 0;
      for (const auto& input : tensorList) {
        auto axis_dim = input.dim32(axis);
        math::CopyMatrix<CPUContext>(
            input.itemsize(),
            before,
            axis_dim * after,
            input.raw_data(),
            axis_dim * after,
            static_cast<char*>(output.raw_mutable_data(input_zero.dtype())) +
                output_offset,
            output_channels * after,
            context,
            input_zero.dtype().copy());
        output_offset += axis_dim * after * input.itemsize();
      }

      return output;
    */
}

#[inline] pub fn stack(
    tensor_list: &Vec<Tensor>,
    axis:        i32,
    context:     *mut CPUContext) -> Tensor 
{
    
    todo!();
    /*
        // 1 - Compute new dimensions
      std::vector<int64_t> newDims(tensorList[0].sizes().vec());
      std::vector<Tensor> expandedTensorList;
      newDims.insert(newDims.begin() + axis, 1);
      for (int i = 0; i < tensorList.size(); i++) {
        expandedTensorList.emplace_back(tensorList[i].Clone());
        expandedTensorList.at(i).Reshape(newDims);
      }
      return cat(expandedTensorList, axis, context);
    */
}

#[inline] pub fn sigmoid(x: &Tensor) -> Tensor {
    
    todo!();
    /*
        Tensor Y(X.sizes(), CPU);
      auto N = X.numel();
      EigenVectorArrayMap<float>(Y.template mutable_data<float>(), N) = 1.0 /
          (1.0 +
           (-ConstEigenVectorArrayMap<float>(X.template data<float>(), N)).exp());
      return Y;
    */
}

#[inline] pub fn tanh(x: &Tensor, context: *mut CPUContext) -> Tensor {
    
    todo!();
    /*
        Tensor Y(X.sizes(), CPU);
      math::Tanh<float, CPUContext>(
          X.numel(),
          X.template data<float>(),
          Y.template mutable_data<float>(),
          context);
      return Y;
    */
}

#[inline] pub fn add(x: &Tensor, y: &Tensor, context: *mut CPUContext) -> Tensor {
    
    todo!();
    /*
        Tensor Z(X.sizes().vec(), CPU);
      math::Add<float, CPUContext>(
          X.numel(),
          X.template data<float>(),
          Y.template data<float>(),
          Z.template mutable_data<float>(),
          context);
      return Z;
    */
}

#[inline] pub fn mul(x: &Tensor, y: &Tensor, context: *mut CPUContext) -> Tensor {
    
    todo!();
    /*
        Tensor Z(X.sizes().vec(), CPU);
      math::Mul<float, CPUContext>(
          X.numel(),
          X.template data<float>(),
          Y.template data<float>(),
          Z.template mutable_data<float>(),
          context);
      return Z;
    */
}

#[inline] pub fn transpose(
    x:       &Tensor, 
    dim0:    i32, 
    dim1:    i32, 
    context: *mut CPUContext) -> Tensor 
{
    todo!();
    /*
        int ndim = X.dim();
      CAFFE_ENFORCE(ndim > dim0 && ndim > dim1, "Invalid transpose dimensions");
      std::vector<int> axes(ndim);
      std::iota(axes.begin(), axes.end(), 0);
      std::swap(axes[dim0], axes[dim1]);
      const std::vector<std::int64_t> X_dims = X.sizes().vec();
      std::vector<std::int64_t> Y_dims(ndim);
      for (int i = 0; i < ndim; ++i) {
        Y_dims[i] = X_dims[axes[i]];
      }
      Tensor Y(Y_dims, CPU);
      math::Transpose<std::int64_t, float, CPUContext>(
          ndim,
          X_dims.data(),
          axes.data(),
          X.template data<float>(),
          Y.template mutable_data<float>(),
          context);
      return Y;
    */
}
