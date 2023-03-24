crate::ix!();

pub const kRealBatchSizeBlob: &'static str = "real_batch_size";
pub const kInitializers: &'static str = "initializers";
pub const kBufferSize: usize = 64;

pub struct OnnxifiTransformer {
    base: BackendTransformerBase,

    /// Options
    opts:              OnnxifiTransformerOptions,

    /// Pointer to loaded onnxifi library
    lib:               *mut OnnxifiLibrary, // default = nullptr

    /// Number of backends
    num_backends:      usize, // default = 0

    /// backend idx
    idx:               i32, // default = 0

    /// Number of Onnxifi Ops we build so far
    onnxifi_op_id:     i32, // default = 0

    /// Model id
    model_id:          String,

    /// Backned IDs
    backend_ids:       Vec<OnnxBackendID>,

    /// A cache for ONNX shape hints
    shape_hints_onnx:  HashMap<String,TensorShape>,

    /// Partition info
    partition_infos:   Vec<PartitionInfo>,
}

impl Drop for OnnxifiTransformer {

    fn drop(&mut self) {
        todo!();
        /* 
          for (unsigned i = 0; i < num_backends_; ++i) {
            if (lib_->onnxReleaseBackendID(backend_ids_[i]) != ONNXIFI_STATUS_SUCCESS) {
              LOG(ERROR) << "Error when calling onnxReleaseBackendID";
            }
          }
         */
    }
}

impl OnnxifiTransformer {
    
    pub fn new(opts: &OnnxifiTransformerOptions) -> Self {
    
        todo!();
        /*
            : BackendTransformerBase(), opts_(opts) 

      lib_ = OnnxinitOnnxifiLibrary();
      CAFFE_ENFORCE(lib_, "Cannot initialize ONNXIFI library");
      CAFFE_ENFORCE_EQ(
          lib_->onnxGetBackendIDs(nullptr, &num_backends_),
          ONNXIFI_STATUS_FALLBACK);
      CAFFE_ENFORCE_GT(
          num_backends_, 0, "At least 1 onnxifi backend should be available");
      backend_ids_.resize(num_backends_);
      CAFFE_ENFORCE_EQ(
          lib_->onnxGetBackendIDs(backend_ids_.data(), &num_backends_),
          ONNXIFI_STATUS_SUCCESS);
        */
    }
    
    /**
      | Check that output shape hints are present
      | to ensure we can pass them to OnnxifiOp
      |
      */
    #[inline] pub fn can_pass_output_shape_hints_per_bs(&self, op: &OperatorDef, shape_hints_per_bs: &HashMap<i32,ShapeInfoMap>) -> bool {
        
        todo!();
        /*
            if (shape_hints_per_bs.empty()) {
        return false;
      }

      for (int bs = 1; bs < opts_.bound_shape_spec.max_batch_size; ++bs) {
        auto shape_hints_search = shape_hints_per_bs.find(bs);
        if (shape_hints_search == shape_hints_per_bs.end()) {
          return false;
        }
        const auto& shape_hints = shape_hints_search->second;

        for (int output_idx = 0; output_idx < op.output_size(); ++output_idx) {
          auto shape_hint_search = shape_hints.find(op.output(output_idx));
          if (shape_hint_search == shape_hints.end()) {
            return false;
          }
        }
      }

      return true;
        */
    }
    
    /**
      | We already have all the ops and external
      | inputs and outputs!
      |
      */
    #[inline] pub fn build_onnxifi_op(&mut self, 
        onnx_model_str:      &String,
        initialization_list: &HashSet<String>,
        external_inputs:     &Vec<String>,
        external_outputs:    &Vec<String>,
        shape_hints_max_bs:  &ShapeInfoMap,
        shape_hints_per_bs:  &HashMap<i32,ShapeInfoMap>) -> OperatorDef {

        todo!();
        /*
            OperatorDef op;
      op.set_type("Onnxifi");
      auto* onnx_model_arg = op.add_arg();
      onnx_model_arg->set_name("onnx_model");
      onnx_model_arg->set_s(onnx_model_str);

      // Add the names of the initializer blobs that we want to fetch from the
      // workspace later
      auto* initializers_arg = op.add_arg();
      initializers_arg->set_name(kInitializers);
      for (const auto& s : initialization_list) {
        initializers_arg->add_strings(s);
      }

      // Add the input/output
      int idx = 0;
      auto* input_names = op.add_arg();
      input_names->set_name("input_names");
      for (const auto& input : external_inputs) {
        if (!initialization_list.count(input)) {
          op.add_input(input);
          input_names->add_strings(input);
        }
      }
      auto* output_names = op.add_arg();
      output_names->set_name("output_names");
      for (const auto& output : external_outputs) {
        op.add_output(output);
        output_names->add_strings(output);
      }

      // Find out the index of input that has a nominal batch size
      const auto max_batch_size = opts_.bound_shape_spec.max_batch_size;
      idx = 0;
      int nominal_batch_idx{0};
      for (const auto& input : external_inputs) {
        if (!initialization_list.count(input)) {
          const auto it = shape_hints_max_bs.find(input);
          CAFFE_ENFORCE(
              it != shape_hints_max_bs.end(),
              "Input shape for ",
              input,
              " not found");
          const auto& info = it->second;
          if (info.getDimType(0) == TensorBoundShape_DimType_BATCH &&
              getBlob1stDimSize(info) == max_batch_size) {
            nominal_batch_idx = idx;
            break;
          }
          ++idx;
        }
      }

      // Add output size hints for max batch size
      auto* output_shape_info_arg = op.add_arg();
      output_shape_info_arg->set_name("output_shape_info");
      auto* output_qshape_info_arg = op.add_arg();
      output_qshape_info_arg->set_name("output_qshape_info");
      for (int i = 0; i < op.output_size(); ++i) {
        const auto& o = op.output(i);
        const auto it = shape_hints_max_bs.find(o);
        if (it != shape_hints_max_bs.end()) {
          if (!it->second.is_quantized) {
            output_shape_info_arg->mutable_tensors()->Add()->CopyFrom(
                wrapShapeInfoIntoTensorProto(o, it->second));
          } else {
            output_qshape_info_arg->mutable_qtensors()->Add()->CopyFrom(
                wrapShapeInfoIntoQTensorProto(o, it->second));
          }
          VLOG(2) << "Adding output hint: " << o;
        }
      }

      // Add output size hints per batch size
      if (canPassOutputShapeHintsPerBs(op, shape_hints_per_bs)) {
        VLOG(2) << "Passing in output shape hints for batch sizes in [1, "
                << opts_.bound_shape_spec.max_batch_size << ")";
        AddArgument("use_passed_output_shapes", 1, &op);

        for (int bs = 1; bs < opts_.bound_shape_spec.max_batch_size; ++bs) {
          auto* output_shape_arg = op.add_arg();
          output_shape_arg->set_name("output_shapes_bs_" + to_string(bs));
          auto* output_qshape_arg = op.add_arg();
          output_qshape_arg->set_name("output_qshapes_bs_" + to_string(bs));

          const auto& shape_hints = shape_hints_per_bs.find(bs)->second;

          for (int output_idx = 0; output_idx < op.output_size(); ++output_idx) {
            const auto& output_name = op.output(output_idx);
            const auto& shape_hint = shape_hints.find(output_name)->second;
            if (!shape_hint.is_quantized) {
              output_shape_arg->mutable_tensors()->Add()->CopyFrom(
                  wrapShapeInfoIntoTensorProto(output_name, shape_hint));
            } else {
              output_shape_arg->mutable_qtensors()->Add()->CopyFrom(
                  wrapShapeInfoIntoQTensorProto(output_name, shape_hint));
            }
          }
        }
      } else {
        AddArgument("use_passed_output_shapes", 0, &op);
      }

      // Tell Onnxifi op that the model is in onnx or c2 proto format
      AddArgument("use_onnx", opts_.use_onnx ? 1 : 0, &op);

      // Tell Onnxifi op which backend id to use
      AddArgument("backend_id", idx_, &op);

      // Add model_id and net_pos to the onnxifi model
      AddArgument(kModelId, model_id_, &op);
      AddArgument(kNetPos, c10::to_string(onnxifi_op_id_++), &op);

      // Add output resizing hints
      if (opts_.adjust_batch) {
        AddArgument("adjust_output_batch", 1, &op);
      } else {
        AddArgument("adjust_output_batch", 0, &op);
      }
      AddArgument("max_batch_size", opts_.bound_shape_spec.max_batch_size, &op);
      AddArgument("max_seq_size", opts_.bound_shape_spec.max_seq_size, &op);
      AddArgument("timeout", opts_.timeout, &op);
      AddArgument("nominal_batch_idx", nominal_batch_idx, &op);

      return op;
        */
    }
    
    /**
      | Convert a cutoff subgraph net to an
      | Onnxifi op
      |
      */
    #[inline] pub fn subnet_to_onnxifi_op_viac2(&mut self, 
        net:                &NetDef,
        weights_in_ws:      &HashSet<String>,
        shape_hints_max_bs: &ShapeInfoMap,
        shape_hints_per_bs: &HashMap<i32,ShapeInfoMap>) -> NetDef {

        todo!();
        /*
            int onnxifi_op_id = onnxifi_op_id_;
      if (opts_.debug) {
        WriteProtoToTextFile(
            net,
            "debug_original_net_" + c10::to_string(onnxifi_op_id) + ".pb_txt",
            false);
      }
      if (opts_.min_ops > net.op_size()) {
        return net;
      }
      // We already have all the ops and external inputs and outputs!
      NetDef onnxifi_net(net);

      // Remove the second output of Concat/Reshape from external_output. Remove
      // rest of the outputs of LayerNorm too. In addition, we remove those outputs
      // from the Onnxifi op too.
      // TODO: This approach is a bit hacky as we assume that the second output is
      // never used. A more appropriate approach can be learned from the ONNX path,
      // where we statically computes the split_info given input shape and insert a
      // GivenTensorIntFill op
      std::unordered_set<std::string> split_infos;
      for (auto& op : *onnxifi_net.mutable_op()) {
        if ((op.type() == "Concat" || op.type() == "Reshape") &&
            op.output_size() == 2) {
          split_infos.emplace(op.output(1));
        } else if (
            op.type() == "SparseLengthsSum" ||
            op.type() == "SparseLengthsSumFused8BitRowwise" ||
            op.type() == "SparseLengthsWeightedSum" ||
            op.type() == "SparseLengthsWeightedSumFused8BitRowwise" ||
            op.type() == "SparseLengthsSumFused4BitRowwise" ||
            op.type() == "SparseLengthsWeightedSumFused4BitRowwise") {
          int weighted = (op.type() == "SparseLengthsWeightedSum" ||
                          op.type() == "SparseLengthsWeightedSumFused8BitRowwise" ||
                          op.type() == "SparseLengthsWeightedSumFused4BitRowwise")
              ? 1
              : 0;
          const auto& indices_hint = shape_hints_max_bs.at(op.input(1 + weighted));
          const auto& lengths_hint = shape_hints_max_bs.at(op.input(2 + weighted));
          const auto& indices_shape = indices_hint.shape;
          const auto& lengths_shape = lengths_hint.shape;
          if ((indices_hint.getDimType(0) ==
                   TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX ||
               indices_hint.getDimType(0) ==
                   TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX_DEFAULT) &&
              indices_shape.dims_size() == 1 && lengths_shape.dims_size() == 1 &&
              indices_shape.dims(0) == lengths_shape.dims(0)) {
            op.add_arg()->CopyFrom(MakeArgument<int>("length1", 1));
          }
        } else if (op.type() == "LayerNorm" && op.output_size() > 1) {
          for (int i = 1; i < op.output_size(); ++i) {
            split_infos.emplace(op.output(i));
          }
        }
      }
      onnxifi_net.clear_external_output();
      for (const auto& o : net.external_output()) {
        if (!split_infos.count(o)) {
          onnxifi_net.add_external_output(o);
        }
      }

      // Figure out weights and add it to external_inputs too
      std::unordered_set<std::string> initialization_list;
      std::vector<std::string> total_inputs_vec;
      getWeightsAndInputs(
          net,
          weights_in_ws,
          std::vector<std::string>(),
          &initialization_list,
          &total_inputs_vec);
      auto* shape_arg = onnxifi_net.add_arg();
      auto* qshape_arg = onnxifi_net.add_arg();
      shape_arg->set_name("input_shape_info");
      qshape_arg->set_name("input_qshape_info");
      std::sort(total_inputs_vec.begin(), total_inputs_vec.end());
      onnxifi_net.clear_external_input();
      for (const auto& i : total_inputs_vec) {
        onnxifi_net.add_external_input(i);
        auto info = shape_hints_max_bs.at(i);
        if (!info.is_quantized) {
          shape_arg->mutable_tensors()->Add()->CopyFrom(
              wrapShapeInfoIntoTensorProto(i, shape_hints_max_bs.at(i)));
        } else {
          qshape_arg->mutable_qtensors()->Add()->CopyFrom(
              wrapShapeInfoIntoQTensorProto(i, shape_hints_max_bs.at(i)));
        }
      }

      // Add partition info
      for (const auto& p : partition_infos_) {
        onnxifi_net.add_partition_info()->CopyFrom(p);
      }

      // Add initializers (weights) list to the net as an arg
      auto* w_arg = onnxifi_net.add_arg();
      w_arg->set_name(kInitializers);
      for (const auto& i : initialization_list) {
        w_arg->add_strings(i);
      }

      // Build ONNXIFI Op
      std::string model_str;
      onnxifi_net.SerializeToString(&model_str);
      std::vector<std::string> onnxifi_net_inputs(
          onnxifi_net.external_input().begin(), onnxifi_net.external_input().end());
      std::vector<std::string> onnxifi_net_outputs(
          onnxifi_net.external_output().begin(),
          onnxifi_net.external_output().end());
      auto onnxifi_op = buildOnnxifiOp(
          model_str,
          initialization_list,
          onnxifi_net_inputs,
          onnxifi_net_outputs,
          shape_hints_max_bs,
          shape_hints_per_bs);
      NetDef net_opt = composeResultNet(onnxifi_op);

      // Debugging stuff
      if (opts_.debug) {
        WriteProtoToTextFile(
            onnxifi_net,
            "debug_onnxifi_net_" + c10::to_string(onnxifi_op_id) + ".pb_txt",
            false);
        WriteProtoToTextFile(
            net_opt,
            "debug_optimized_net_" + c10::to_string(onnxifi_op_id) + ".pb_txt",
            false);
      }
      return net_opt;
        */
    }
    
    /**
      | Since we create new tensors during the
      | conversion process, we actually need into
      | inject them into the original workspace
      |
      | Since our onnx exporter uses
      | std::unordered_map<std::string,
      | TensorShape> as lut, we need to include an
      | extra copy of shape info and maintain them
      | together
      */
    #[inline] pub fn subnet_to_onnxifi_op_via_onnx(&mut self, 
        net:                &NetDef,
        weights_in_ws:      &HashSet<String>,
        ws:                 *mut Workspace,
        exporter:           *mut OnnxExporter,
        shape_hints_max_bs: *mut ShapeInfoMap,
        shape_hints_per_bs: &HashMap<i32,ShapeInfoMap>) -> NetDef {
        
        todo!();
        /*
            if (opts_.min_ops > net.op_size()) {
        return net;
      }
      ::ONNX_NAMESPACE::ModelProto onnx_model;
      fillModelInfo(&onnx_model);

      NetDef onnxifi_net(net);

      // Convert c2 ops to onnx ops, add const weights if there are any
      DeviceOption option;
      CPUContext context(option);
      context.SwitchToDevice();
      std::vector<std::string> extra_weights;
      for (const auto& op : onnxifi_net.op()) {
        const auto results = exporter->Caffe2OpToOnnxNodes(op, shape_hints_onnx_);
        for (const auto& n : results.first) {
          onnx_model.mutable_graph()->add_node()->CopyFrom(n);
        }
        for (const auto& t : results.second) {
          VLOG(2) << "Adding extra init tensor: " << t.name();
          TensorShape shape;
          shape.mutable_dims()->CopyFrom(t.dims());
          auto ret = shape_hints_onnx_.emplace(t.name(), std::move(shape));
          shape_hints_max_bs->emplace(
              std::piecewise_construct,
              std::forward_as_tuple(ret.first->first),
              std::forward_as_tuple(
                  std::vector<TensorBoundShape::DimType>(
                      shape.dims_size(), TensorBoundShape_DimType_CONSTANT),
                  ret.first->second));

          // Feed into workspace as CPU Tensors
          auto* blob = ws->CreateBlob(t.name());
          auto* cpu_tensor = BlobGetMutableTensor(blob, CPU);
          std::vector<int64_t> dims;
          for (const auto& d : t.dims()) {
            dims.push_back(d);
          }
          cpu_tensor->Resize(dims);
          if (t.data_type() == ::ONNX_NAMESPACE::TensorProto::FLOAT) {
            context.CopyBytesSameDevice(
                cpu_tensor->numel() * sizeof(float),
                static_cast<const void*>(t.raw_data().data()),
                cpu_tensor->raw_mutable_data(TypeMeta::Make<float>()));
          } else if (t.data_type() == ::ONNX_NAMESPACE::TensorProto::INT64) {
            context.CopyBytesSameDevice(
                cpu_tensor->numel() * sizeof(int64_t),
                static_cast<const void*>(t.raw_data().data()),
                cpu_tensor->raw_mutable_data(TypeMeta::Make<int64_t>()));
          } else {
            CAFFE_THROW(
                "Unsupported tensor data type for conversion: ", t.data_type());
          }
          context.FinishDeviceComputation();

          // Add mappings
          extra_weights.emplace_back(t.name());
        }
      }

      // Convert outputs and compute output shape hints
      std::vector<std::string> onnxifi_net_outputs;
      for (const auto& o : net.external_output()) {
        onnxifi_net_outputs.emplace_back(o);
      }
      auto io_vec = convertToValueInfo(
          onnxifi_net_outputs,
          shape_hints_onnx_,
          std::unordered_map<std::string, ::ONNX_NAMESPACE::TypeProto>());
      for (const auto& i : io_vec) {
        onnx_model.mutable_graph()->add_output()->CopyFrom(i);
      }

      // Convert inputs and figure out weights
      std::unordered_set<std::string> initialization_list;
      std::vector<std::string> onnxifi_net_inputs;
      getWeightsAndInputs(
          net,
          weights_in_ws,
          extra_weights,
          &initialization_list,
          &onnxifi_net_inputs);
      io_vec = convertToValueInfo(
          onnxifi_net_inputs,
          shape_hints_onnx_,
          std::unordered_map<std::string, ::ONNX_NAMESPACE::TypeProto>());
      for (const auto& i : io_vec) {
        onnx_model.mutable_graph()->add_input()->CopyFrom(i);
      }

      // Onnx model is ready. Build ONNXIFI Op
      std::string model_str;
      onnx_model.SerializeToString(&model_str);
      auto onnxifi_op = buildOnnxifiOp(
          model_str,
          initialization_list,
          onnxifi_net_inputs,
          onnxifi_net_outputs,
          *shape_hints_max_bs,
          shape_hints_per_bs);
      NetDef net_opt = composeResultNet(onnxifi_op);

      // Debugging stuff
      if (opts_.debug) {
        WriteProtoToTextFile(onnx_model, "debug_onnxifi_net.onnx_txt", false);
        WriteProtoToTextFile(net_opt, "debug_optimized_net.pb_txt", false);
      }
      return net_opt;
        */
    }
    
    /**
      | Query whether an operator is supported
      | by passing ONNX protobuf
      |
      */
    #[inline] pub fn support_op_onnx(&self, 
        op:              &OperatorDef,
        exporter:        *mut OnnxExporter,
        blocklisted_ops: &HashSet<i32>,
        backend_id:      OnnxBackendID) -> bool {
        
        todo!();
        /*
            try {
        int pos =
            ArgumentHelper::GetSingleArgument<OperatorDef, int>(op, kNetPos, -1);
        if (blocklisted_ops.count(pos)) {
          LOG(INFO) << "Skipping blocklisted op " << op.type() << " at pos " << pos;
          return false;
        }
        const OpSchema* schema = OpSchemaRegistry::Schema(op.type());
        // NB: this might not be a hard constraint as we can just export C2
        // domain specific ops to ONNX
        if (!schema || schema->onnx_schema().empty()) {
          LOG(INFO) << "Cannot export c2 op " << op.type()
                    << " to onnx as there is no corresponding ONNX schema.";
          return false;
        }

        ::ONNX_NAMESPACE::ModelProto onnx_model;
        fillModelInfo(&onnx_model);
        auto results = exporter->Caffe2OpToOnnxNodes(op, shape_hints_onnx_);
        std::unordered_set<std::string> used_inputs;
        std::unordered_set<std::string> used_outputs;
        std::vector<std::string> boundary_inputs;
        std::vector<std::string> boundary_outputs;
        std::unordered_set<std::string> reshape_info;
        // nodes are in topological order, so we just need to iterate
        for (const auto& n : results.first) {
          onnx_model.mutable_graph()->add_node()->CopyFrom(n);
          for (const auto& i : n.input()) {
            bool is_new = used_inputs.emplace(i).second;
            // The input is not seen and it's not referred by any nodes before as
            // output, we count it as an boundary input
            if (is_new && !used_outputs.count(i)) {
              boundary_inputs.emplace_back(i);
            }
          }
          for (const auto& o : n.output()) {
            used_outputs.emplace(o);
          }

          // For reshape node, if it has more than 1 inputs, we need to feed the
          // second input which contains the shape info
          if (n.op_type() == "Reshape" && n.input_size() > 1) {
            reshape_info.emplace(n.input(1));
          }
        }
        // Second iteration to account all the boundary outputs, which is a newly
        // seen output and is not referred as input before
        used_outputs.clear();
        for (const auto& n : results.first) {
          for (const auto& o : n.output()) {
            bool is_new = used_outputs.emplace(o).second;
            if (is_new && !used_inputs.count(o)) {
              boundary_outputs.emplace_back(o);
            }
          }
        }
        std::unordered_map<std::string, ::ONNX_NAMESPACE::TypeProto>
            extra_shape_hints;
        for (const auto& t : results.second) {
          extra_shape_hints.emplace(t.name(), OnnxExtraTypeProto(t));
          if (reshape_info.count(t.name())) {
            onnx_model.mutable_graph()->add_initializer()->CopyFrom(t);
          }
        }

        // Add input/output shape info
        auto io_vec = convertToValueInfo(
            boundary_inputs, shape_hints_onnx_, extra_shape_hints);
        for (const auto& i : io_vec) {
          onnx_model.mutable_graph()->add_input()->CopyFrom(i);
        }
        io_vec = convertToValueInfo(
            boundary_outputs, shape_hints_onnx_, extra_shape_hints);
        for (const auto& i : io_vec) {
          onnx_model.mutable_graph()->add_output()->CopyFrom(i);
        }

        std::string onnx_model_str;
        onnx_model.SerializeToString(&onnx_model_str);
        auto ret = lib_->onnxGetBackendCompatibility(
            backend_id, onnx_model_str.size(), onnx_model_str.c_str());
        if (ret != ONNXIFI_STATUS_SUCCESS) {
          LOG(INFO) << "Don't support onnx for " << op.type() << " c2 op (" << ret
                    << ")";
          return false;
        } else {
          return true;
        }
      } catch (const std::exception& ex) {
        LOG(ERROR) << "Caught exception when converting op " << op.type()
                   << ", what: " << ex.what();
        return false;
      }
        */
    }
    
    /**
      | Query whether an operator is supported
      | by passing C2 protobuf
      |
      */
    #[inline] pub fn support_opc2(&self, 
        op:              &OperatorDef,
        shape_hints:     &ShapeInfoMap,
        weights:         &HashSet<String>,
        blocklisted_ops: &HashSet<i32>,
        backend_id:      OnnxBackendID) -> bool {

        todo!();
        /*
            try {
        int pos =
            ArgumentHelper::GetSingleArgument<OperatorDef, int>(op, kNetPos, -1);
        if (blocklisted_ops.count(pos)) {
          LOG(INFO) << "Skipping blocklisted op " << op.type() << " at pos " << pos;
          return false;
        }

        // Build a c2 net with one op
        NetDef net;
        net.add_op()->CopyFrom(op);
        std::unordered_set<std::string> seenExternalInputs;
        for (const auto& i : op.input()) {
          if (seenExternalInputs.count(i)) {
            continue;
          }
          seenExternalInputs.insert(i);
          net.add_external_input(i);
        }
        for (const auto& o : op.output()) {
          net.add_external_output(o);
        }
        // Remove the second output of Concat/Reshape from the external_output
        if ((op.type() == "Concat" || op.type() == "Reshape") &&
            op.output_size() == 2) {
          net.mutable_external_output()->RemoveLast();
        } else if (op.type() == "LayerNorm" && op.output_size() > 1) {
          int remove = op.output_size() - 1;
          for (int i = 0; i < remove; ++i) {
            net.mutable_external_output()->RemoveLast();
          }
        }

        // Encode the input/output shapes to an argument
        auto* shape_arg = net.add_arg();
        auto* qshape_arg = net.add_arg();
        shape_arg->set_name("input_shape_info");
        qshape_arg->set_name("input_qshape_info");
        std::unordered_set<std::string> seenInputsForShapeArgs;
        for (const auto& i : op.input()) {
          if (seenInputsForShapeArgs.count(i)) {
            continue;
          }
          seenInputsForShapeArgs.insert(i);
          const auto it = shape_hints.find(i);
          if (it == shape_hints.end()) {
            VLOG(1) << "Skipping " << op.type() << " (" << pos
                    << ") due to missing shape info for input " << i;
            return false;
          }
          if ((it->second).is_quantized == false) {
            shape_arg->mutable_tensors()->Add()->CopyFrom(
                wrapShapeInfoIntoTensorProto(i, it->second));
          } else {
            qshape_arg->mutable_qtensors()->Add()->CopyFrom(
                wrapShapeInfoIntoQTensorProto(i, it->second));
          }
        }

        qshape_arg = net.add_arg();
        shape_arg = net.add_arg();
        shape_arg->set_name("output_shape_info");
        qshape_arg->set_name("output_qshape_info");
        for (const auto& i : op.output()) {
          const auto it = shape_hints.find(i);
          if (it == shape_hints.end()) {
            VLOG(1) << "Skipping " << op.type() << " (" << pos
                    << ") due to missing shape info for output " << i;
            return false;
          }
          if ((it->second).is_quantized == false) {
            shape_arg->mutable_tensors()->Add()->CopyFrom(
                wrapShapeInfoIntoTensorProto(i, it->second));
          } else {
            qshape_arg->mutable_qtensors()->Add()->CopyFrom(
                wrapShapeInfoIntoQTensorProto(i, it->second));
          }
        }

        // Annnote the inputs that are weights
        auto w_arg = net.add_arg();
        w_arg->set_name(kInitializers);
        for (const auto& i : op.input()) {
          if (weights.count(i)) {
            w_arg->add_strings(i);
          }
        }

        std::string c2_model_str;
        net.SerializeToString(&c2_model_str);
        auto ret = lib_->onnxGetBackendCompatibility(
            backend_id, c2_model_str.size(), c2_model_str.c_str());
        if (ret != ONNXIFI_STATUS_SUCCESS) {
          LOG(INFO) << "Don't support c2 op " << op.type() << " at pos " << pos
                    << " (" << ret << ")";
          return false;
        } else {
          return true;
        }
      } catch (const std::exception& ex) {
        LOG(ERROR) << "Caught exception when converting op " << op.type()
                   << ", what: " << ex.what();
        return false;
      }
        */
    }
    
    /**
      | Tie the output of Gather to the scalar
      | weight input of the SparseLengthsWeighted*
      | and SparseLengthsSumSparseLookup (which is
      | split from the
      | SparseLengthsWeighted*Sparse) ops. If the
      | latter is disabled, disable the former
      | too.
      */
    #[inline] pub fn tie_gather_and_sparse_lengths_weighted_sum_ops(&self, 
        net:             &NetDef,
        shape_hints:     &ShapeInfoMap,
        weights:         &HashSet<String>,
        blocklisted_ops: *mut HashSet<i32>)  {

        todo!();
        /*
            std::unordered_map<std::string, int> output_pos;
      OnnxExporter exporter(nullptr);
      onnxBackendID backend_id = backend_ids_[idx_];

      for (const auto& op : net.op()) {
        std::string check;
        if (op.type() == "Gather") {
          int pos =
              ArgumentHelper::GetSingleArgument<OperatorDef, int>(op, kNetPos, -1);
          for (const auto& output : op.output()) {
            output_pos.emplace(output, pos);
          }
        } else if (StartsWith(op.type(), "SparseLengthsWeighted")) {
          auto supported = opts_.use_onnx
              ? supportOpOnnx(op, &exporter, *blocklisted_ops, backend_id)
              : supportOpC2(op, shape_hints, weights, *blocklisted_ops, backend_id);
          if (!supported && op.input_size() > 1) {
            check = op.input(1);
          }
        } else if (
            op.type() == "SparseLengthsSumSparseLookup" && op.input_size() > 3) {
          check = op.input(3);
        }
        if (!check.empty()) {
          const auto it = output_pos.find(check);
          if (it == output_pos.end()) {
            continue;
          }
          blocklisted_ops->emplace(it->second);
          // We know that current op is not going to be supported. Might as well
          // blocklist it too
          blocklisted_ops->emplace(
              ArgumentHelper::GetSingleArgument<OperatorDef, int>(op, kNetPos, -1));
        }
      }
        */
    }
    
    /**
      | For net with partitioning info, blocklist
      | ops that are supposed to run on CPU, whose
      | partition info will contain empty
      | device_id list.
      */
    #[inline] pub fn blocklist_cpu_partition(&self, net: &NetDef, blocklisted_ops: *mut HashSet<i32>)  {
        
        todo!();
        /*
            std::unordered_set<std::string> cpu_partitions;
      for (const auto& p : partition_infos_) {
        if (p.device_id_size() == 0) {
          cpu_partitions.emplace(p.name());
        }
      }
      for (const auto& op : net.op()) {
        const auto& pname = op.device_option().node_name();
        if (cpu_partitions.count(pname)) {
          blocklisted_ops->emplace(
              ArgumentHelper::GetSingleArgument<OperatorDef, int>(op, kNetPos, -1));
        }
      }
        */
    }
    
    /// Rule based filtering
    #[inline] pub fn apply_filtering_rules(&self, 
        net:             &NetDef,
        shape_hints:     &ShapeInfoMap,
        weights:         &HashSet<String>,
        blocklisted_ops: *mut HashSet<i32>)  {

        todo!();
        /*
            tieGatherAndSparseLengthsWeightedSumOps(
          net, shape_hints, weights, blocklisted_ops);
      blocklistCpuPartition(net, blocklisted_ops);
        */
    }
    
    /// Determine backend id
    #[inline] pub fn get_backend_id(&mut self) -> Vec<OnnxBackendID> {
        
        todo!();
        /*
            idx_ = 0;

      if (opts_.use_onnx) {
        return backend_ids_;
      }
      // Try to find a backend that support Caffe2 proto. Note that this is quite
      // opportunistic as we don't officially support Caffe2 proto.
      char buf[kBufferSize];
      for (int i = 0; i < backend_ids_.size(); ++i) {
        size_t len = kBufferSize;
        auto ret = lib_->onnxGetBackendInfo(
            backend_ids_[i], ONNXIFI_BACKEND_DEVICE, buf, &len);
        if (ret == ONNXIFI_STATUS_SUCCESS && strstr(buf, "Caffe2")) {
          LOG(INFO) << "Using backend with Caffe2 Proto, ID: " << i;
          idx_ = i;
          break;
        }
      }
      return backend_ids_;
        */
    }
    
    /// Transform by passing C2 proto to backend
    #[inline] pub fn transform_viac2(&mut self, 
        pred_net:           *mut NetDef,
        weights:            &HashSet<String>,
        blocklisted_ops:    &HashSet<i32>,
        shape_hints_max_bs: &ShapeInfoMap,
        shape_hints_per_bs: &HashMap<i32,ShapeInfoMap>) -> NetDef {

        todo!();
        /*
            onnxBackendID backend_id = backend_ids_[idx_];

      auto c2_supports =
          [this, &shape_hints_max_bs, &blocklisted_ops, backend_id, &weights](
              const OperatorDef& op) {
            return supportOpC2(
                op, shape_hints_max_bs, weights, blocklisted_ops, backend_id);
          };

      auto c2_converter = [this,
                           &weights,
                           &shape_hints_max_bs,
                           &shape_hints_per_bs](const NetDef& net) {
        return SubnetToOnnxifiOpViaC2(
            net, weights, shape_hints_max_bs, shape_hints_per_bs);
      };

      return opt::OptimizeForBackend(
          *pred_net, c2_supports, c2_converter, opts_.debug);
        */
    }
    
    /// Transform by passing ONNX proto to backend
    #[inline] pub fn transform_via_onnx(&mut self, 
        ws:                 *mut Workspace,
        pred_net:           *mut NetDef,
        weights:            &HashSet<String>,
        blocklisted_ops:    &HashSet<i32>,
        shape_hints_max_bs: *mut ShapeInfoMap,
        shape_hints_per_bs: &HashMap<i32,ShapeInfoMap>) -> NetDef {

        todo!();
        /*
            onnxBackendID backend_id = backend_ids_[idx_];

      // function to tell whether the ONNXIFI backend supports a given C2 op or not
      OnnxExporter exporter(nullptr);
      auto onnx_supports = [this, &exporter, &blocklisted_ops, backend_id](
                               const OperatorDef& op) {
        return supportOpOnnx(op, &exporter, blocklisted_ops, backend_id);
      };

      // function to convert runnable subgraph into an onnxifi op. We need to keep
      // the same exporter throughout the process to avoid duplicated dummy name
      // generation
      OnnxExporter exporter2(nullptr);
      auto onnx_converter = [this,
                             ws,
                             &weights,
                             shape_hints_max_bs,
                             &exporter2,
                             &shape_hints_per_bs](
                                const NetDef& net) mutable {
        return SubnetToOnnxifiOpViaOnnx(
            net, weights, ws, &exporter2, shape_hints_max_bs, shape_hints_per_bs);
      };

      return opt::OptimizeForBackend(
          *pred_net, onnx_supports, onnx_converter, opts_.debug);
        */
    }
    
    /**
      | Extract partition info from the original
      | net
      |
      */
    #[inline] pub fn extract_partition_info(&mut self, net: &NetDef)  {
        
        todo!();
        /*
            partition_infos_.clear();
      for (const auto& p : net.partition_info()) {
        partition_infos_.emplace_back(p);
      }
        */
    }
    
    /**
      | Cutting off the runnable part and replace
      | with ONNXIFI ops. Asssume the nets were
      | topologically sorted
      |
      */
    #[inline] pub fn transform(&mut self, 
        ws:                *mut Workspace,
        pred_net:          *mut NetDef,
        weight_names:      &Vec<String>,
        input_shape_hints: &ShapeInfoMap,
        blocklisted_ops:   &HashSet<i32>)  {

        todo!();
        /*
            CAFFE_ENFORCE(ws);
      CAFFE_ENFORCE(pred_net, "Predict net cannot be nullptr");

      if (opts_.debug) {
        WriteProtoToTextFile(*pred_net, "debug_pre_ssa_net.pb_txt", false);
      }

      // Get model id and reset Onnxifi op id to 0
      model_id_ = getModelId(*pred_net);
      onnxifi_op_id_ = 0;

      // Unroll If ops
      fetchInputsToIfOpsSubnet(pred_net);

      std::unordered_set<std::string> weights(
          weight_names.begin(), weight_names.end());

      // SSA Rewrite the net if it has not been rewritten
      ShapeInfoMap shape_hints_mapped;
      if (opts_.predictor_net_ssa_rewritten) {
        LOG(INFO) << "predictor net has been ssaRewritten, skip rewritting here";
        annotateOpIndex(pred_net);
        shape_hints_mapped = input_shape_hints;
        for (const auto& w : weights) {
          input_mapping_.emplace(w, w);
        }
      } else {
        shape_hints_mapped = ssaRewriteAndMapNames(ws, pred_net, input_shape_hints);
      }

      // Populate shape info
      // TODO(yingz): We should not need to create mapped_ws since we did not change
      // any input mappings during ssarewrite. However this is here for the
      // following reason: BlackBoxPredictor calls RunNetOnce before onnxifi to
      // populate dimension info. However during this, it was observed, that new
      // blob for output is created. This causes problem if inferShape uses original
      // ws since it does not expect the output blob to be present.
      Workspace mapped_ws(ws, input_mapping_);
      ShapeInfoMap shape_hints_max_bs = inferShapes(
          &mapped_ws, pred_net, shape_hints_mapped, opts_.bound_shape_spec);
      if (opts_.use_onnx) {
        shape_hints_onnx_ = stripShapeInfoMap(shape_hints_max_bs);
      }
      if (opts_.enforce_fp32_inputs_into_fp16) {
        enforceFp32InputsToFp16(weights, pred_net, &shape_hints_max_bs);
      }
      if (opts_.merge_fp32_inputs_into_fp16) {
        mergeFp32InputsAndConvertToFp16(
            opts_.bound_shape_spec.max_batch_size,
            weights,
            pred_net,
            &shape_hints_max_bs);
      }

      if (opts_.debug) {
        NetDef ssa_net;
        ssa_net.CopyFrom(*pred_net);
        auto* w_arg = ssa_net.add_arg();
        w_arg->set_name(kInitializers);
        for (const auto& w : weights) {
          w_arg->add_strings(w);
        }
        dumpNet(ssa_net, shape_hints_max_bs, "debug_ssa_net.pb_txt");
      }
      extractPartitionInfo(*pred_net);

      // Get backend id
      getBackendId();

      // Apply some filtering rules
      std::unordered_set<int> new_blocklisted_ops(
          blocklisted_ops.begin(), blocklisted_ops.end());
      applyFilteringRules(
          *pred_net, shape_hints_max_bs, weights, &new_blocklisted_ops);

      // Transform the net
      NetDef net_opt = opts_.use_onnx ? TransformViaOnnx(
                                            ws,
                                            pred_net,
                                            weights,
                                            new_blocklisted_ops,
                                            &shape_hints_max_bs,
                                            opts_.shape_hints_per_bs)
                                      : TransformViaC2(
                                            pred_net,
                                            weights,
                                            new_blocklisted_ops,
                                            shape_hints_max_bs,
                                            opts_.shape_hints_per_bs);

      // Need to figure out a proper place to handle device option
      net_opt.mutable_device_option()->CopyFrom(pred_net->device_option());
      net_opt.set_type(pred_net->type());

      pred_net->Swap(&net_opt);

      addShapeToNet(*pred_net, shape_hints_max_bs);
      if (opts_.debug) {
        WriteProtoToTextFile(*pred_net, "debug_full_opt_net.pb_txt", false);
      }
        */
    }
}
