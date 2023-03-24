crate::ix!();

#[inline] pub fn dump_graph(g: *mut NNGraph, fname: &String)  {
    
    todo!();
    /*
        auto nnprinter = [](typename NNGraph::NodeRef node) {
        std::map<std::string, std::string> labelMap;
        assert(node->data() && "Node doesn't have data, can't render it");
        if (isa<NeuralNetOperator>(node->data())) {
          auto* op = dyn_cast<NeuralNetOperator>(node->data().get());
          const auto& op_def =
              dyn_cast<Caffe2Annotation>(op->getAnnotation())->getOperatorDef();
          int pos = -1;
          for (const auto& arg : op_def.arg()) {
            if (arg.name() == "net_pos") {
              if (arg.has_i()) {
                pos = arg.i();
              }
              break;
            }
          }
          labelMap["label"] =
              op->getName() + " (" + c10::to_string((unsigned long long)node) + ")";
          auto* annotation = op->getAnnotation();
          if (annotation && isa<Caffe2Annotation>(annotation)) {
            auto device_annotation = dyn_cast<Caffe2Annotation>(annotation);
            labelMap["label"] += "\\n[" + device_annotation->getDevice() +
                ", pos=" + c10::to_string(pos) + "]";
            auto hash = std::hash<std::string>{}(device_annotation->getDevice());
            std::stringstream hex_stream;
            hex_stream << std::hex << hash;
            labelMap["color"] = "#" + hex_stream.str().substr(0, 6);
            labelMap["fontcolor"] = labelMap["color"];
          }
          labelMap["shape"] = "box";
        } else if (isa<Data>(node->data())) {
          auto tensor = dyn_cast<NeuralNetData>(node->data().get());
          labelMap["label"] = tensor->getName();
          labelMap["label"] += "_" + c10::to_string(tensor->getVersion()) + " " +
              c10::to_string((unsigned long long)node);
        }
        return labelMap;
      };

      std::ofstream out(fname.c_str());
      if (out) {
        out << nom::converters::convertToDotString(g, nnprinter);
      } else {
        LOG(ERROR) << "Cannot create nomnigraph dump file: " << fname;
      }
    */
}
