crate::ix!();

#[inline] pub fn merge_external_tensors<T,U>(
    curr_external: &HashSet<NodeRef<T,U>>, 
    old_external: &Vec<String>) -> Vec<String> {

    todo!();
    /*
        std::vector<std::string> out;

      // Maximally preserve the order of external inputs and outputs.
      std::unordered_set<std::string> newExternal;
      for (const auto& tensorNode : currExternal) {
        CAFFE_ENFORCE(
            repr::nn::is<repr::NeuralNetData>(tensorNode),
            "A non-tensor node was added to external inputs/outputs of the NNModule");
        auto name = repr::nn::get<repr::NeuralNetData>(tensorNode)->getName();
        newExternal.insert(name);
      }

      for (const auto& tensorName : oldExternal) {
        if (newExternal.count(tensorName)) {
          out.emplace_back(tensorName);
          newExternal.erase(tensorName);
        }
      }
      for (const auto& tensorName : newExternal) {
        out.emplace_back(tensorName);
      }

      return out;
    */
}
