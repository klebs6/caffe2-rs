crate::ix!();

/**
  | @brief
  | 
  | Convert to an NNModule and apply a mapping
  | of tensor names to DeviceOptions to
  | it.
  | 
  | This *only* applies the map to Declare/Export
  | nodes, which are representationally
  | equivalent to external_input/external_output
  | in caffe2 NetDefs.
  | 
  | Throws an exception if the passed in
  | blobMap contains blobs that are not
  | present in the NNModule.
  |
  */
#[inline] pub fn convert_to_nnmodule<T,U>(
    net:      &mut NetDef, 
    blob_map: HashMap<String,DeviceOption>) -> NNModule<T,U> 
{
    todo!();
    /*
        auto nn = convertToNNModule(net);
      injectDataEdgeIndicators(&nn);
      addBlobDeviceOptions(blobMap, &nn);
      return nn;
    */
}

#[inline] pub fn convert_to_caffe_2proto<T,U>(m: &mut NNModule<T,U>) -> NetDef {
    
    todo!();
    /*
        auto predictNet = caffe2::NetDef();
      return convertToCaffe2Proto(m, predictNet);
    */
}

/**
  | Pass in an oldNet to copy all the attributes
  | of that network.
  |
  | Be warned that transformations that modify the
  | graph's inputs or outputs are not reflected in
  | changes to external_input or external_output.
  */
#[inline] pub fn convert_to_caffe_2proto_with_old_net<T,U>(
    m: &mut NNModule<T,U>, 
    old_net: &NetDef) -> NetDef 
{
    todo!();
    /*
        auto predictNet = caffe2::NetDef();
      // We copy the old net rather than mutate it.
      predictNet.CopyFrom(oldNet);
      predictNet.mutable_op()->Clear();

      repr::nn::coalesceInsertedDataDependencies(&m);

      // Simply iterate through the CFG and populate data dependencies
      // with the DFG
      for (const auto& bbNode : m.controlFlow.getMutableNodes()) {
        if (bbNode->getOutEdges().size() > 1) {
          CAFFE_THROW("Control flow not yet supported in Caffe2 converter.");
        }
        auto& bb = bbNode->data();
        for (const auto& instrNode : bb.getInstructions()) {
          caffe2::OperatorDef op = convertToOperatorDef(instrNode);

          for (const auto& inEdge : instrNode->getInEdges()) {
            auto* tensorNode =
                dyn_cast<repr::NeuralNetData>(inEdge->tail()->data().get());
            *op.add_input() = tensorNode->getName();
          }
          for (const auto& outEdge : instrNode->getOutEdges()) {
            auto* tensorNode =
                dyn_cast<repr::NeuralNetData>(outEdge->head()->data().get());
            *op.add_output() = tensorNode->getName();
          }

          auto* nnOp = repr::nn::get<repr::NeuralNetOperator>(instrNode);
          if (nnOp->getLayout() != repr::NeuralNetOperator::NNLayout::Undefined) {
            caffe2::Argument* arg = nullptr;
            for (int i = 0; i < op.arg_size(); ++i) {
              auto arg_ = op.mutable_arg(i);
              if (arg_->name() == "order") {
                arg = arg_;
                break;
              }
            }

            if (!arg) {
              arg = op.add_arg();
              arg->set_name("order");
            }

            auto layout = nnOp->getLayout();
            if (layout == repr::NeuralNetOperator::NNLayout::NCHW) {
              arg->set_s("NCHW");
            }
            if (layout == repr::NeuralNetOperator::NNLayout::NHWC) {
              arg->set_s("NHWC");
            }
          }

          // Save the operator to the net.
          *predictNet.add_op() = op;
        }
      }

      // Maximally preserve the order of external inputs and outputs.
      std::vector<std::string> oldExternalInputs;
      std::vector<std::string> oldExternalOutputs;

      for (const auto& inputName : predictNet.external_input()) {
        oldExternalInputs.emplace_back(inputName);
      }
      for (const auto& outputName : predictNet.external_output()) {
        oldExternalOutputs.emplace_back(outputName);
      }

      auto newExternalInputs = mergeExternalTensors(m.inputs, oldExternalInputs);
      auto newExternalOutputs = mergeExternalTensors(m.outputs, oldExternalOutputs);

      predictNet.clear_external_input();
      predictNet.clear_external_output();

      for (const auto& inputName : newExternalInputs) {
        predictNet.add_external_input(inputName);
      }

      for (const auto& outputName : newExternalOutputs) {
        predictNet.add_external_output(outputName);
      }

      return predictNet;
    */
}
