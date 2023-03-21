crate::ix!();

/**
  | Retrieve the top-K elements of the last
  | dimension.
  | 
  | Given an input tensor of shape $(a_1,
  | a_2, ..., a_n, r)$. `k` can be passed
  | as an integer argument or a 1D tensor
  | containing a single integer.
  | 
  | Returns up to three outputs:
  | 
  | 1. Value tensor of shape $(a_1, a_2,
  | ..., a_n, k)$ which contains the values
  | of the top k elements along the last dimension
  | 
  | 2. Index tensor of shape $(a_1, a_2,
  | ..., a_n, k)$ which contains the indices
  | of the top k elements (original indices
  | from the input tensor).
  | 
  | 3. [OPTIONAL] Flattened index tensor
  | of shape $(a_1 * a_2 * ... * a_n * k,)$.
  | 
  | Given two equivalent values, this operator
  | uses the indices along the last dimension
  | as a tiebreaker.
  | 
  | That is, the element with the lower index
  | will appear first.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/top_k.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct TopKOp<T,Context> {
    context: Context,
    k:       i32,
    axis:    i32,
    phantom: PhantomData<T>,
}

num_inputs!{TopK, (1,2)}

num_outputs!{TopK, (2,3)}

inputs!{TopK, 
    0 => ("X", "(*Tensor`<float>`*): input tensor of shape $(a_1, a_2, ..., a_n, r)$"),
    1 => ("k", "(*int*): number of top elements to retrieve")
}

outputs!{TopK, 
    0 => ("Values",             "(*Tensor`<float>`*): output tensor of shape $(a_1, a_2, ..., a_n, k)$"),
    1 => ("Indices",            "(*Tensor`<int>`*): tensor of indices of shape $(a_1, a_2, ..., a_n, k)$; indices values refer to each element's index in the last dimension of the `X` input tensor"),
    2 => ("Flattened_indices",  "(*Tensor`<int>`*): tensor of indices of shape $(a_1 * a_2 * ... * a_n * k,)$; indices values refer to each element's index in the flattened input tensor `X`")
}

register_cpu_operator!{
    TopK, 
    TopKOp<float, CPUContext>
}

tensor_inference_function!{
    TopK, 
    /* [](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out = {in[0], in[0]};
      ArgumentHelper helper(def);
      auto k = helper.GetSingleArgument("k", -1);
      auto dims_size = in[0].dims_size();
      out[0].set_dims(dims_size - 1, k);
      out[1].set_dims(dims_size - 1, k);
      out[1].set_data_type(TensorProto_DataType_INT32);
      if (def.output_size() > 2) {
        TensorShape flatten_indices_shape;
        flatten_indices_shape.set_data_type(TensorProto_DataType_INT32);
        flatten_indices_shape.add_dims(
            std::accumulate(
                in[0].dims().begin(),
                in[0].dims().end() - 1,
                1,
                std::multiplies<long>()) *
            k);
        out.push_back(flatten_indices_shape);
      }
      return out;
    } */
}
