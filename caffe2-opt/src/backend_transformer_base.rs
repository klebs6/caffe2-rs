crate::ix!();

/**
  | This class contains some common functions
  | for backend lowering and graph cutting
  |
  */
pub struct BackendTransformerBase {

    /**
      | Input mapping of input name -> original
      | input name
      |
      */
    input_mapping:         HashMap<String,String>,

    /**
      | Input mapping of original input name
      | -> input name
      |
      */
    reverse_input_mapping: HashMap<String,String>,
}

impl BackendTransformerBase {
    
    #[inline] pub fn input_mapping(&self) -> &HashMap<String,String> {
        
        todo!();
        /*
            return input_mapping_;
        */
    }
    
    #[inline] pub fn reverse_input_mapping(&self) -> &HashMap<String,String> {
        
        todo!();
        /*
            return reverse_input_mapping_;
        */
    }
    
    /**
      | SSA rewrite the net and return name mapping
      |
      */
    #[inline] pub fn ssa_rewrite_and_map_names(
        &mut self, 
        ws:                *mut Workspace,
        pred_net:          *mut NetDef,
        input_shape_hints: &ShapeInfoMap) -> ShapeInfoMap 
    {
        todo!();
        /*
          input_mapping_ = onnx::SsaRewrite(nullptr, pred_net);
          // Annote the ops with net position
          annotateOpIndex(pred_net);

          // Since we are going to create a mapped workspace, we need to make sure that
          // the parent workspace has the mapped blob names. If the blobs don't exist
          // (usually such blobs are input tensor names), we exclude them from mapping.
          std::vector<std::string> exclude_mapping;
          for (const auto kv : input_mapping_) {
            if (!ws->HasBlob(kv.second)) {
              exclude_mapping.emplace_back(kv.first);
            }
          }
          for (const auto& i : exclude_mapping) {
            input_mapping_.erase(i);
          }

          ShapeInfoMap shape_hints_mapped;
          for (const auto& kv : input_shape_hints) {
            shape_hints_mapped.emplace(kv.first, kv.second);
          }
          return shape_hints_mapped;
        */
    }
    
    /**
      | Do bound shape inference and collect
      | shape infos
      |
      */
    #[inline] pub fn infer_shapes(
        &mut self, 
        ws:                 *mut Workspace,
        pred_net:           *mut NetDef,
        shape_hints_mapped: &ShapeInfoMap,
        spec:               &BoundShapeSpec) -> ShapeInfoMap 
    {
        todo!();
        /*
            ShapeInfoMap shape_map;

      // Populate shapes from workplace
      const std::vector<std::string> ws_blobs = ws->Blobs();
      for (const auto& s : ws_blobs) {
        auto shape_info = getShapeInfoFromBlob(ws->GetBlob(s));
        if (shape_info.dimTypeIsSet()) {
          shape_map.emplace(s, shape_info);
        }
      }
      for (const auto& s : shape_hints_mapped) {
        shape_map.insert(s);
      }
      auto eng = BoundShapeInferencerRegistry()->Create("C10", spec);
      eng->InferBoundShapeAndType(*pred_net, shape_map, ws);
      const auto& out_map = eng->shape_info();
      shape_map.clear();
      for (const auto& kv : out_map) {
        shape_map.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(kv.first),
            std::forward_as_tuple(
                kv.second.getDimType(),
                kv.second.shape,
                kv.second.is_quantized,
                kv.second.q_info));
      }
      return shape_map;
        */
    }
    
    /// add shape info to the net
    #[inline] pub fn add_shape_to_net(
        &self, 
        shape_net: &mut NetDef, 
        shape_hints: &ShapeInfoMap)  
    {
        todo!();
        /*
          auto* shape_arg = shape_net.add_arg();
          auto* qshape_arg = shape_net.add_arg();
          shape_arg->set_name("shape_info");
          qshape_arg->set_name("qshape_info");
          for (const auto& kv : shape_hints) {
            if (!kv.second.is_quantized) {
              auto t = wrapShapeInfoIntoTensorProto(kv.first, kv.second);
              shape_arg->mutable_tensors()->Add()->CopyFrom(t);
            } else {
              auto t = wrapShapeInfoIntoQTensorProto(kv.first, kv.second);
              qshape_arg->mutable_qtensors()->Add()->CopyFrom(t);
            }
          }
        */
    }

    /// Dump the net with shape info
    #[inline] pub fn dump_net(
        &self, 
        pred_net: &NetDef,
        shape_hints: &ShapeInfoMap,
        fname: &String)  {

        todo!();
        /*
           NetDef shape_net(pred_net);
           addShapeToNet(shape_net, shape_hints);
           WriteProtoToTextFile(shape_net, fname, false);
           */
    }

    /// Get model ID from the NetDef
    #[inline] pub fn get_model_id(net: &NetDef) -> String {
        
        todo!();
        /*
            static std::atomic<size_t> seq_id{0};
          std::string model_id;
          for (const auto& arg : net.arg()) {
            if (arg.name() == kModelId) {
              if (arg.has_s()) {
                model_id = arg.s();
              } else if (arg.has_i()) {
                model_id = c10::to_string(arg.i());
              }
              break;
            }
          }

          if (model_id.empty()) {
            model_id = "unnamed_" + c10::to_string(seq_id++);
          }
          return model_id;
        */
    }

    /**
      | Populate 'net_pos' argument for any ops
      | that don't already have it. 'net_pos' we
      | populate here starts after the max
      | 'net_pos' value we encountered.
      */
    #[inline] pub fn annotate_op_index(net: *mut NetDef)  {
        
        todo!();
        /*
            // find the max net_pos that we have so far.
          int i = -1;
          for (const auto& op : net->op()) {
            ArgumentHelper helper(op);
            int old_index = helper.GetSingleArgument(op, kNetPos, -1);
            i = std::max(i, old_index);
          }

          // populate net_pos for any op that doesn't already have it.
          for (auto& op : *(net->mutable_op())) {
            if (!ArgumentHelper::HasArgument(op, kNetPos)) {
              AddArgument(kNetPos, ++i, &op);
            }
          }
        */
    }
}
