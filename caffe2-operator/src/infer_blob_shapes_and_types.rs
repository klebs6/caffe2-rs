crate::ix!();

#[inline] pub fn infer_blob_shapes_and_types(
    blob_desc: &mut HashMap<String, TensorShape>,
    nets:      &Vec<*mut NetDef>)  
{
    todo!();
    /*
        for (auto& defptr : nets) {
        // Hack to work with auto split gradients
        CaffeMap<string, string> unmatched_sum_blobs;
        CaffeMap<string, TensorShape> reshape_cache;
        CaffeMap<string, vector<TensorShape>> split_cache;

        for (const OperatorDef& op : defptr->op()) {
          // Hack to ignore queues
          if (op.type().find("Dequeue") != std::string::npos ||
              op.type().find("Enqueue") != std::string::npos) {
            continue;
          }

          vector<TensorShape> input_desc;
          bool found_all = true;
          for (const string& in : op.input()) {
            auto inp_desc = blob_desc.find(in);
            if (inp_desc == blob_desc.end()) {
              LOG(WARNING) << "Shape and type inference failed for input: " << in
                           << " for op " << op.type() << ", skipping.";
              found_all = false;
              break;
            }
            input_desc.push_back(inp_desc->second);
          }
          if (!found_all) {
            continue;
          }
          auto op_schema = OpSchemaRegistry::Schema(op.type());
          if (op_schema == nullptr) {
            LOG(WARNING) << "Shape inference failed, no schema for: " << op.type();
            continue;
          }

          // Special handling for Sum as it used with the autosplits, which have
          // different naming convention. Assuming that all sum inputs must be of
          // same size, we can infer their shapes.
          if (op.type() == "Sum") {
            TensorShape sum_shape;
            for (auto inp : op.input()) {
              auto it = blob_desc.find(inp);
              if (it != blob_desc.end() && !it->second.unknown_shape()) {
                if (it->second.dims_size() > 0) {
                  sum_shape = blob_desc[inp];
                  break;
                }
              }
            }
            for (auto inp : op.input()) {
              auto it = blob_desc.find(inp);
              if (it == blob_desc.end() || it->second.unknown_shape()) {
                blob_desc[inp] = sum_shape;
                if (sum_shape.dims_size() == 0) {
                  // Match later with the output
                  unmatched_sum_blobs[inp] = op.output(0);
                }
              }
            }
          }

          if (op.type() == "Reshape" && op.is_gradient_op()) {
            CAFFE_ENFORCE(reshape_cache.find(op.input(1)) != reshape_cache.end());
            TensorShape cached = reshape_cache[op.input(1)];
            blob_desc[op.output(0)] = cached;
            TensorShape dims;
            dims.add_dims(cached.dims_size());
            dims.set_data_type(TensorProto_DataType_INT64);
            blob_desc[op.output(1)] = dims;
            continue;
          } else if (
              op.type() == "Split" && op.input_size() == 2 && op.is_gradient_op()) {
            CAFFE_ENFORCE(split_cache.find(op.input(1)) != split_cache.end());
            vector<TensorShape> cached = split_cache[op.input(1)];
            CAFFE_ENFORCE_EQ(op.output_size(), cached.size());
            for (size_t i = 0; i < cached.size(); i++) {
              blob_desc[op.output(i)] = cached[i];
            }
            continue;
          }

          std::vector<TensorShape> out;
          try {
            out = op_schema->InferTensor(op, input_desc);
            if (op.is_gradient_op() && out.size()) {
              // Special handling for gradient ops. We can assume gradients
              // are of same size as the corresponding variables. This is bit
              // ugly to base on string matching, but we don't have the connection
              // between variable and its gradient specified

              CaffeMap<string, string> grads_to_params =
                  GradientMakerBase::MatchGradsToParams(op);

              for (size_t i = 0; i < out.size(); i++) {
                if (out[i].unknown_shape()) {
                  std::string gradout = op.output(i);

                  if (grads_to_params.find(gradout) != grads_to_params.end()) {
                    std::string var = grads_to_params[gradout];
                    if (blob_desc.find(var) != blob_desc.end()) {
                      out[i] = blob_desc[var];
                    }
                  }
                }
              }
            }

            if (op.type() == "Reshape") {
              // Reshape stores the original input shape to its second output
              // blob. We need this for gradient reshape.
              reshape_cache[op.output(1)] = input_desc[0];
            } else if (op.type() == "Concat") {
              // Split needs the input sizes from Concat.
              split_cache[op.output(1)] = input_desc;
            }

          } catch (::caffe2::EnforceNotMet& enf) {
            LOG(ERROR) << "Shape inference error: " << enf.what();
            LOG(ERROR) << "Operator: " << ProtoDebugString(op) << std::endl;
            LOG(ERROR) << "Returning empty results.";

            TensorShapes tps;
            return tps;
          }

          if (out.size() != (unsigned)op.output_size()) {
            if (op.type() == "Slice") {
              CAFFE_ENFORCE(
                  out.size() == 0,
                  "For Slice operator, either shape of all output blobs are "
                  "inferred or shape of none can be inferred.");
            } else {
              CAFFE_THROW(
                  "Invalid shape inference for operator ",
                  op.type(),
                  " Expected ",
                  op.output_size(),
                  " outputs, but got ",
                  out.size());
            }
          } else {
            for (size_t i = 0; i < out.size(); i++) {
              blob_desc[op.output(i)] = out[i];
            }
          }
        } // net.ops

        for (auto& unmatched : unmatched_sum_blobs) {
          if (blob_desc.find(unmatched.second) != blob_desc.end()) {
            blob_desc[unmatched.first] = blob_desc[unmatched.second];
          }
        }

      } // nets
      TensorShapes tps;
      for (auto kv : blob_desc) {
        TensorShape& tp = kv.second;
        TensorShape* tpnew = tps.add_shapes();
        tpnew->CopyFrom(tp);
        tpnew->set_name(kv.first);
      }
      return tps;
    */
}

