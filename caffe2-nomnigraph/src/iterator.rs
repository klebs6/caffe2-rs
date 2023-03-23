crate::ix!();

pub fn node_iterator<T, U, G: GraphType>(g: &mut G) -> Vec<<G as GraphType>::NodeRef> {

    todo!();
    /*
      std::vector<typename G::NodeRef> out;
      for (auto node : g.getMutableNodes()) {
        if (!is<T>(node)) {
          continue;
        }
        out.emplace_back(node);
      }
      return out;
    */
}

#[inline] pub fn filter<T, U>(nn: &mut NNModule<T,U>) -> Vec<NodeRef<T,U>> {

    todo!();
    /*
        return nodeIterator<T>(nn.dataFlow);
    */
}

pub fn data_iterator<T, U, G: GraphType>(g: &mut G) -> Vec<(*mut T, <G as GraphType>::NodeRef)> {

    todo!();
    /*
      std::vector<std::pair<T*, typename G::NodeRef>> out;
      for (auto node : g.getMutableNodes()) {
        if (!is<T>(node)) {
          continue;
        }
        auto d = get<T>(node);
        out.emplace_back(std::make_pair(d, node));
      }
      return out;
    */
}

