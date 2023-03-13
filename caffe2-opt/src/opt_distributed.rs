crate::ix!();


#[inline] pub fn set_device_option(
    n: NNGraph_NodeRef, 
    d: &mut DeviceOption)  {
    
    todo!();
    /*
        getOrAddCaffe2Annotation(n);
      auto op = nn::get<NeuralNetOperator>(n);
      auto c2Annot = dyn_cast<caffe2::Caffe2Annotation>(op->getMutableAnnotation());
      CAFFE_ENFORCE(c2Annot, "getOrAddCaffe2Annotation failed!");
      c2Annot->setDeviceOption(d);
    */
}

/**
  | Helpers for the convertToNNModule for use if
  | you already have an NNModule.
  |
  | You probably don't want to use these if you
  | can use convertToNNModule instead.
  */
#[inline] pub fn add_blob_device_options<T,U>(
    blob_map: HashMap<String,DeviceOption>, 
    nn:       *mut NNModule<T,U>)  
{
    todo!();
    /*
        // Names we've seen in the NNModule
      std::unordered_set<std::string> seen;

      auto declareNodes = nn::filter<Declare>(*nn);

      for (auto& declareNode : declareNodes) {
        auto inputNode = nn::getOutputs(declareNode).at(0);
        auto input = nn::get<nom::repr::Tensor>(inputNode);

        if (!blobMap.count(input->getName())) {
          continue;
        }

        CAFFE_ENFORCE(
            !seen.count(input->getName()),
            "Ambiguous name->deviceOption map.  Please do this manually.");

        seen.insert(input->getName());
        setDeviceOption(declareNode, blobMap[input->getName()]);
      }

      auto exportNodes = nn::filter<Export>(*nn);

      for (auto& exportNode : exportNodes) {
        auto outputNode = nn::getInputs(exportNode).at(0);
        auto output = nn::get<nom::repr::Tensor>(outputNode);

        if (!blobMap.count(output->getName())) {
          continue;
        }

        CAFFE_ENFORCE(
            !seen.count(output->getName()),
            "Ambiguous name->deviceOption map.  Please do this manually.");

        seen.insert(output->getName());
        setDeviceOption(exportNode, blobMap[output->getName()]);
      }

      if (seen.size() != blobMap.size()) {
        std::ostringstream os;
        for (const auto& kv : blobMap) {
          if (!(seen.count(kv.first))) {
            os << "\"" << kv.first << "\" ";
          }
        }
        CAFFE_ENFORCE(
            seen.size() == blobMap.size(),
            "Unused names in the blob map: ",
            os.str());
      }
    */
}

#[inline] pub fn inject_data_edge_indicators<T,U>(nn: *mut NNModule<T,U>)  {
    
    todo!();
    /*
        for (auto& input : nn->inputs) {
        auto declareNode =
            nn->dataFlow.createNode(std::make_unique<Declare>());
        nn->dataFlow.createEdge(declareNode, input);
      }

      for (auto& output : nn->outputs) {
        auto exportNode = nn->dataFlow.createNode(std::make_unique<Export>());
        nn->dataFlow.createEdge(output, exportNode);
      }

      nn->inputs.clear();
      nn->outputs.clear();
    */
}

#[inline] pub fn remove_data_edge_indicators<T,U>(nn: *mut NNModule<T,U>)  {
    
    todo!();
    /*
        auto declareNodes = nn::filter<Declare>(*nn);
      for (auto& declareNode : declareNodes) {
        auto input = nn::getOutputs(declareNode).at(0);
        nn->inputs.insert(input);
        nn->dataFlow.deleteNode(declareNode);
      }
      auto exportNodes = nn::filter<Export>(*nn);
      for (auto& exportNode : exportNodes) {
        auto output = nn::getInputs(exportNode).at(0);
        nn->outputs.insert(output);
        nn->dataFlow.deleteNode(exportNode);
      }
    */
}

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
