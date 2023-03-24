crate::ix!();

/**
  | \brief Ingest a caffe2 protobuf model and
  | output an NNModule.
  |
  | \param net The caffe2 protobuf NetDef
  |
  | Default conversion to a NNModule
  |
  | Optionally strict -- which checks for various
  | input and output conditions.
  |
  | Optionally this function will update a vector
  | that maps operators in the netdef positionally
  | to NodeRefs in the resultant NNModule.
  */
#[inline] pub fn convert_to_nnmodule<T,U>(
    net:         &NetDef,
    strict:      Option<bool>,
    op_node_vec: *mut Vec<NodeRef<T,U>>) -> NNModule<T,U> 
{
    let strict = strict.unwrap_or(false);
    
    todo!();
    /*
        repr::NNModule module;
      repr::NNGraph& dfg = module.dataFlow;
      repr::NNCFGraph& cfg = module.controlFlow;
      /// \brief We keep track of the producer of the blob.
      /// Because Caffe2 Nets are really just ordered operations
      /// we can just keep track of the most recent producer of
      /// a blob and draw and edge from that to any consumer we
      /// come by. If a new operator produces the blob we simply
      /// replace it in this map.
      std::unordered_map<std::string, repr::NNGraph::NodeRef> blobMap;

      std::unordered_set<std::string> externalInputNames;
      for (const auto& inputName : net.external_input()) {
        externalInputNames.insert(inputName);
      }

      /// \brief For the construction of the control flow graph we keep track
      /// of a current basic block, which we split up as we come across control
      /// flow operations such as if and while.
      auto bbNode = cfg.createNamedFunction("main");

      for (const auto& op : net.op()) {
        auto opNode = dfg.createNode(); // Create an empty node for the operator.
        // First calculate in-edges (data dependencies).
        for (const auto& input : op.input()) {
          // If we've never seen this tensor, make one.
          if (!blobMap.count(input)) {
            auto tensor = std::make_unique<repr::Tensor>(input);
            blobMap[input] =
                dfg.createNode(unique_dyn_cast<repr::NeuralNetData>(tensor));
            if (externalInputNames.count(input)) {
              module.inputs.insert(blobMap[input]);
              externalInputNames.erase(input);
            }
          }

          auto tensorNode = blobMap[input];
          dfg.createEdge(tensorNode, opNode);
        }

        // Then save outputs into the blobMap for later consumption.
        for (const auto& output : op.output()) {
          auto tensor = std::make_unique<repr::Tensor>(output);
          auto tensorNode =
              dfg.createNode(unique_dyn_cast<repr::NeuralNetData>(tensor));
          dfg.createEdge(opNode, tensorNode);
          blobMap[output] = tensorNode;
        }

        opNode->resetData(convertToNeuralNetOperator(op));
        if (opNodeVec) {
          opNodeVec->emplace_back(opNode);
        }
        auto currentBasicBlock = bbNode->mutableData();
        currentBasicBlock->pushInstructionNode(opNode);
      }

      if (externalInputNames.size()) {
        // In strict mode we ensure the input names are valid
        if (strict) {
          std::ostringstream os;
          for (const auto& inputName : externalInputNames) {
            os << "\"" << inputName << "\" ";
          }

          CAFFE_ENFORCE(
              externalInputNames.size() == 0,
              "Attempting to convert an ill-formed network: ",
              "external_input contains ",
              externalInputNames.size(),
              " unused blobs: ",
              os.str());
          // Otherwise, we add the blobs to the graph as no-ops
        } else {
          for (const auto& input : externalInputNames) {
            blobMap[input] = dfg.createNode(std::make_unique<repr::Tensor>(input));
          }
        }
      }

      for (const auto& outputName : net.external_output()) {
        CAFFE_ENFORCE(
            !strict || blobMap.count(outputName),
            "NetDef has ill-formed external_output:",
            outputName);
        if (!blobMap.count(outputName)) {
          LOG(ERROR) << "NetDef has ill-formed external_output: " << outputName;
          continue;
        }
        module.outputs.insert(blobMap[outputName]);
      }

      return module;
    */
}


