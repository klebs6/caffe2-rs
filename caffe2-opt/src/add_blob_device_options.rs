crate::ix!();

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
