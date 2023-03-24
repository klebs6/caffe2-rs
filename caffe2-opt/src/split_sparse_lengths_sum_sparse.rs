crate::ix!();

/**
  | Split SparseLengthsSumSparse into
  | 
  | SparseLengthsSumSparseLookup + SparseLengthsSum
  |
  */
#[inline] pub fn split_sparse_lengths_sum_sparse(net: *mut NetDef, ws: &Workspace)  {
    
    todo!();
    /*
        const static std::unordered_map<string, string> slss = {
          {"SparseLengthsSum4BitRowwiseSparse", "SparseLengthsSumFused4BitRowwise"},
          {"SparseLengthsWeightedSum4BitRowwiseSparse",
           "SparseLengthsWeightedSumFused4BitRowwise"},
          {"SparseLengthsSum8BitRowwiseSparse", "SparseLengthsSumFused8BitRowwise"},
          {"SparseLengthsWeightedSum8BitRowwiseSparse",
           "SparseLengthsWeightedSumFused8BitRowwise"},
          {"SparseLengthsSum2BitRowwiseSparse", "SparseLengthsSumFused2BitRowwise"},
          {"SparseLengthsWeightedSum2BitRowwiseSparse",
           "SparseLengthsWeightedSumFused2BitRowwise"}};
      NetDef new_net;
      new_net.CopyFrom(*net);
      new_net.mutable_op()->Clear();
      for (const auto& op : net->op()) {
        const auto it = slss.find(op.type());
        if (it == slss.end()) {
          new_net.add_op()->CopyFrom(op);
        } else {
          const bool is_weighted =
              (op.type().find("Weighted") != std::string::npos);
          const auto& compressed_mapping = op.input(is_weighted ? 4 : 3);
          const auto* b = ws.GetBlob(compressed_mapping);
          bool fallback = false;
          if (b && b->IsType<Tensor>()) {
            const auto& t = BlobGetTensor(*b, CPU);
            fallback = ((t.numel() == 1) && (t.template data<int32_t>()[0] == 0));
          }

          if (fallback) {
            // If fallback, we just replace the original slss op with a normal sls
            // op
            OperatorDef new_op;
            new_op.CopyFrom(op);
            new_op.set_type(it->second);
            new_op.mutable_input()->RemoveLast();
            new_net.add_op()->CopyFrom(new_op);
          } else {
            // Otherwise, we replace slss with slss_lookup followed by a normal sls
            OperatorDef new_op;
            new_op.CopyFrom(op);
            new_op.set_type("SparseLengthsSumSparseLookup");
            new_op.clear_input();
            const auto& indices_in = is_weighted ? op.input(2) : op.input(1);
            const auto& lengths_in = is_weighted ? op.input(3) : op.input(2);
            const auto& compress_mapping = is_weighted ? op.input(4) : op.input(3);
            const auto& weights_in = is_weighted ? op.input(1) : "";
            new_op.add_input(indices_in);
            new_op.add_input(lengths_in);
            new_op.add_input(compress_mapping);
            const auto indices_out = indices_in + "_decomp";
            const auto lengths_out = lengths_in + "_decomp";
            const auto weights_out = weights_in + "_decomp";
            new_op.clear_output();
            new_op.add_output(indices_out);
            new_op.add_output(lengths_out);
            if (is_weighted) {
              new_op.add_input(weights_in);
              new_op.add_output(weights_out);
            }
            new_net.add_op()->CopyFrom(new_op);

            new_op.CopyFrom(op);
            new_op.set_type(it->second);
            new_op.mutable_input()->RemoveLast();
            *new_op.mutable_input()->Mutable(is_weighted ? 2 : 1) = indices_out;
            *new_op.mutable_input()->Mutable(is_weighted ? 3 : 2) = lengths_out;
            if (is_weighted) {
              *new_op.mutable_input()->Mutable(1) = weights_out;
            }
            new_net.add_op()->CopyFrom(new_op);
          }
        }
      }

      new_net.Swap(net);
    */
}
