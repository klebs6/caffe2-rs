crate::ix!();

tensor_inference_function!{MatMul, /* [](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      out[0].set_data_type(in[0].data_type());
      ArgumentHelper arg_helper(def);
      int axis_a = arg_helper.GetSingleArgument<int>("axis_a", 1);
      int axis_b = arg_helper.GetSingleArgument<int>("axis_b", 1);
      int trans_a = arg_helper.GetSingleArgument<bool>("trans_a", false);
      int trans_b = arg_helper.GetSingleArgument<bool>("trans_b", false);
      int canonical_axis_a = canonical_axis_index_(axis_a, in[0].dims().size());
      int canonical_axis_b = canonical_axis_index_(axis_b, in[0].dims().size());

      int M = size_to_dim_(canonical_axis_a, GetDimsVector(in[0]));
      int N = size_from_dim_(canonical_axis_b, GetDimsVector(in[1]));
      if (trans_a) {
        M = size_from_dim_(canonical_axis_a, GetDimsVector(in[0]));
      }
      if (trans_b) {
        N = size_to_dim_(canonical_axis_b, GetDimsVector(in[1]));
      }

      out[0].add_dims(M);
      out[0].add_dims(N);

      return out;
    } */
}
