crate::ix!();

#[inline] pub fn prune_unreferered_nodes<T,U>(nn: *mut NNModule<T,U>)  {
    
    todo!();
    /*
        auto& g = nn->dataFlow;
      std::vector<NodeRef> to_delete;
      for (auto node : g.getMutableNodes()) {
        if (!nn::hasProducer(node) && !nn::hasConsumer(node)) {
          to_delete.push_back(node);
        }
      }
      for (auto i : to_delete) {
        if (nn::is<NeuralNetData>(i)) {
          auto name = nn::get<NeuralNetData>(i)->getName();
          auto it = nn->inputs.find(i);
          if (it != nn->inputs.end()) {
            VLOG(2) << "Removing external input " << name;
            nn->inputs.erase(it);
          }
          it = nn->outputs.find(i);
          if (it != nn->outputs.end()) {
            VLOG(2) << "Removing external output " << name;
            nn->outputs.erase(it);
          }
        }
        g.deleteNode(i);
      }
    */
}

