crate::ix!();

#[inline] pub fn pair_larger_than<A, B>(x: &(A,B), y: &(A,B)) -> bool {

    todo!();
    /*
        return x.second > y.second;
    */
}

#[inline] pub fn prune(node_idx: i32, nodes: &mut Vec<OpGraphNode>)  {
    
    todo!();
    /*
        // Ancestor table for tracking the visited nodes
      std::vector<bool> ancestors(nodes.size(), false);
      // stack element is pair of <curr_node, previous_node>
      std::stack<std::pair<int, int>> nodes_stack;
      // initialize the prev_node to be -1
      nodes_stack.push(std::make_pair(node_idx, -1));

      while (!nodes_stack.empty()) {
        const auto& node_pair = nodes_stack.top();
        int curr = node_pair.first;
        int prev = node_pair.second;

        // If the node has already been visited, pop curr out of
        // stack and clean up the ancestor table
        CAFFE_ENFORCE(curr < (int)ancestors.size(), "Out of bound access");
        if (ancestors[curr]) {
          ancestors[curr] = false;
          nodes_stack.pop();
          continue;
        }

        // Check if this has a parent that can be pruned:
        //  if parent is not the previous node visited and is
        //  an ancestor of the current traversar, it can be
        //  pruned.
        if (prev >= 0) {
          std::vector<int> new_parents;
          for (auto parent : nodes[curr].parents_) {
            if (parent != prev && ancestors[parent]) {
              // We can prune this one
              nodes[parent].children_.erase(
                  std::remove(
                      nodes[parent].children_.begin(),
                      nodes[parent].children_.end(),
                      curr),
                  nodes[parent].children_.end());
            } else {
              new_parents.push_back(parent);
            }
          }
          nodes[curr].parents_ = new_parents;
        }

        ancestors[curr] = true;

        // Descend -- but only once from each node
        if (nodes[curr].visited_inputs == nodes[curr].num_orig_parents) {
          const auto& children = nodes[curr].children_;
          for (auto child : children) {
            nodes[child].visited_inputs++;
            nodes_stack.push(std::make_pair(child, curr));
          }
        }
      }
    */
}

/**
  | Prune redundant dependencies to improve
  | chaining.
  | 
  | TODO: t15868555 This algorithm is fast
  | but can miss dependencies.
  |
  */
#[inline] pub fn prune_op_node_graph(nodes: &Vec<OperatorNode>) -> Vec<OpGraphNode> {
    
    todo!();
    /*
        Timer t;
      std::vector<OpGraphNode> pruned;

      // Create a separate list of pruned operatornodes used
      // for the chaining computation. Because of the unique_ptr
      // in the OperatorNode, we cannot do a copy but have to
      // copy just the fields we need.
      for (auto& node : nodes) {
        OpGraphNode nd;
        nd.children_ = node.children_;
        nd.parents_ = node.parents_;
        nd.num_orig_parents = nd.parents_.size();
        pruned.push_back(nd);
      }

      for (int i = 0; i < (int)pruned.size(); ++i) {
        if (pruned[i].parents_.size() == 0) {
          prune(i, pruned);
        }
      }

      LOG(INFO) << "Operator graph pruning prior to chain compute took: "
                << t.Seconds() << " secs";
      return pruned;
    */
}
