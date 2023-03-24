crate::ix!();

/**
  | Onnxifi transformation on the net and
  | workspace.  We also needed the input
  | data/shape to populate the shape. In addition,
  | we take a \p blocklist to control and mask
  | what ops we want to consider in onnxifi
  | process. We can also set whether to use ONNX
  | proto or C2 proto through ONNXIFI interface.
  |
  */
#[inline] pub fn onnxifi(
    net:                         *mut NetDef,
    ws:                          *mut Workspace,
    input_names:                 &Vec<String>,
    output_names:                &Vec<String>,
    weight_names:                &Vec<String>,
    blocklist:                   &HashSet<i32>,
    shape_hints_max_bs:          &ShapeInfoMap,
    use_onnx:                    bool,
    max_batch_size:              Option<usize>,
    max_seq_size:                Option<usize>,
    load_model_by_blob:          Option<bool>,
    predictor_net_ssa_rewritten: Option<bool>,
    shape_hints_per_bs:          Option<&HashMap<i32,ShapeInfoMap>>,
    blacklist_ops:               Option<&String>,
    min_ops:                     Option<usize>)  
{
    // Carrying out the ONNXIFI transform

    let max_batch_size:               usize                     = max_batch_size.unwrap_or(0);
    let max_seq_size:                 usize                     = max_seq_size.unwrap_or(0);
    let load_model_by_blob:           bool                      = load_model_by_blob.unwrap_or(false);
    let predictor_net_ssa_rewritten:  bool                      = predictor_net_ssa_rewritten.unwrap_or(false);

    todo!();
    /*
        // Split SparseLengthsSumSparse so that we can lower the SparseLengthsSum part
      splitSparseLengthsSumSparse(net, *ws);

      // Clean up the external input/output of the net
      net->mutable_external_input()->Clear();
      net->mutable_external_output()->Clear();
      for (const auto& i : input_names) {
        net->add_external_input(i);
      }
      for (const auto& w : weight_names) {
        net->add_external_input(w);
      }
      for (const auto& o : output_names) {
        net->add_external_output(o);
      }

      // ONNXIFI transform
      OnnxifiTransformerOptions opts;
      opts.use_onnx = use_onnx;
      opts.bound_shape_spec.max_batch_size = max_batch_size;
      opts.bound_shape_spec.max_seq_size = max_seq_size;
      opts.debug = FLAGS_onnxifi_debug_mode;
      opts.adjust_batch = FLAGS_onnxifi_adjust_batch;
      opts.min_ops = min_ops.value_or(FLAGS_onnxifi_min_ops);
      opts.load_model_by_blob = load_model_by_blob;
      opts.enforce_fp32_inputs_into_fp16 = FLAGS_enforce_fp32_inputs_into_fp16;
      opts.merge_fp32_inputs_into_fp16 = FLAGS_merge_fp32_inputs_into_fp16;
      opts.predictor_net_ssa_rewritten = predictor_net_ssa_rewritten;
      opts.timeout = FLAGS_onnxifi_timeout_ms;
      opts.shape_hints_per_bs = shape_hints_per_bs;

      ShapeInfoMap more_shape_hints = shape_hints_max_bs;
      if (!FLAGS_onnxifi_shape_hints.empty()) {
        parseShapeInfoMapFromString(FLAGS_onnxifi_shape_hints, more_shape_hints);
      }

      // Before applying backlist, make sure the ops in the net all have an net_pos;
      caffe2::BackendTransformerBase::annotateOpIndex(net);

      // Parse the blocklist
      auto more_blocklist = ParseNetPositionList(FLAGS_onnxifi_blacklist);
      for (const auto& b : blocklist) {
        more_blocklist.emplace(b);
      }

      // ONNX mode will change the op order so it doesn't apply here
      if (!opts.use_onnx) {
        auto blocklisted_ops = ParseBlockListOps(blacklist_ops.value_or(FLAGS_onnxifi_blacklist_ops));
        for (const auto& op : net->op()) {
          if (blocklisted_ops.count(op.type())) {
            ArgumentHelper helper(op);
            more_blocklist.emplace(helper.GetSingleArgument(op, kNetPos, -1));
          }
        }
      }

      // Attach observation nodes
      //
      // When we want to observe intermediate tensors value out of the onnxifi op,
      // we use the following trick:
      //
      // 1. for specified op, we find its input and outputs.
      // 2. for each input and output, we create a new copy op and attach it as an
      // input to the copy.
      // 3. we blocklist these new copy operators from onnxification. This forces
      // these intermediate tensors to also become outputs of the onnxifi op.
      // 4. we put the right arguments on the copy ops so TensorObserver can print
      // out the values.
      auto ops_to_observe =
          ParseNetPositionList(FLAGS_onnxifi_input_output_observe_list);
      std::unordered_set<std::string> tensors_to_observe;
      for (const auto& op : ops_to_observe) {
        if (op >= net->op().size()) {
          CAFFE_THROW(
              "Cannot observe operator at position ", op, " (out of range)");
        }
        const auto& op_to_observe = net->op(op);
        tensors_to_observe.insert(
            op_to_observe.input().begin(), op_to_observe.input().end());

        if ((op_to_observe.type() == "Concat" ||
             op_to_observe.type() == "Reshape") &&
            op_to_observe.output().size() == 2) {
          tensors_to_observe.insert(op_to_observe.output(0));
        } else {
          tensors_to_observe.insert(
              op_to_observe.output().begin(), op_to_observe.output().end());
        }
      }
      for (const auto& tensor : tensors_to_observe) {
        OperatorDef copy_op;
        copy_op.set_type("Copy");
        copy_op.add_input(tensor);
        copy_op.add_output(tensor + "_copy_output_ignore");
        auto pos = net->op().size();
        AddArgument(kNetPos, pos, &copy_op);
        AddArgument("observe_input_tensors", 1, &copy_op);
        net->add_op()->CopyFrom(copy_op);
        more_blocklist.emplace(pos);
      }

      OnnxifiTransformer ts(opts);
      ts.transform(ws, net, weight_names, more_shape_hints, more_blocklist);

      // Cleanup the input from the workspace
      for (const auto& i : input_names) {
        ws->RemoveBlob(i);
      }
    */
}
