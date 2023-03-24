crate::ix!();

#[inline] pub fn compute_chains(orig_nodes: &mut Vec<OperatorNode>) -> ExecutionChains {
    
    todo!();
    /*
        const std::vector<OpGraphNode> nodes = pruneOpNodeGraph(orig_nodes);
      vector<int> initial_frontier;
      for (int idx = 0; idx < (int)nodes.size(); ++idx) {
        if (nodes[idx].parents_.size() == 0) {
          initial_frontier.push_back(idx);
        }
      }

      // We need to construct the node_seen_count to know how many inner edges each
      // node has.
      std::unordered_map<int, int> node_seen_count;

      for (int root_index : initial_frontier) {
        const auto& root = nodes[root_index];
        std::stack<std::pair<int, std::vector<int>::const_iterator>> depth_stack;
        depth_stack.push(make_pair(root_index, root.children_.begin()));
        node_seen_count[root_index]++;
        CAFFE_ENFORCE(
            node_seen_count[root_index] == 1,
            "root node ",
            root_index,
            " visit count must be == 1");

        while (depth_stack.size() > 0) {
          auto cur = depth_stack.top();
          depth_stack.pop();
          if (cur.second != nodes[cur.first].children_.end()) {
            int node_index = *cur.second;
            node_seen_count[node_index]++;
            cur.second++;
            depth_stack.push(cur);
            if (node_seen_count[node_index] == 1) {
              // Visit each child only once.
              depth_stack.push(
                  make_pair(node_index, nodes[node_index].children_.begin()));
            }
          }
        }
      }
      // Now, we compute the set of execution chains An execution chain is
      // a linear set of nodes that can be executed on a single stream
      // (e.g. a chain of single input, single output operators)
      ExecutionChains chains;
      std::unordered_set<int> seen_nodes;
      std::vector<int> chain;
      std::pair<int, std::vector<int>::const_iterator> cur;
      std::stack<std::pair<int, std::vector<int>::const_iterator>> depth_stack;
      auto check_current_for_chaining = [&]() -> bool {
        return (
            node_seen_count[cur.first] == 1 &&
            (chain.size() == 0 ||
             (
                 // A chain of operators is executed without additional
                 // synchronization by calling RunAsync sequentially on each
                 // operator and passing the same stream id on each call.
                 // RunAsync may schedule an async computation on device.
                 // In order to be scheduled on the same chain two operators
                 // (parent and dependent) need to satisfy:
                 //  1. Both ops are on the same device _and_
                 //  2. Parent op does not have an async part or
                 //     dependent op can be executed as an async dependency

                 IsSameDevice(
                     orig_nodes[cur.first].operator_->device_option(),
                     orig_nodes[chain.back()].operator_->device_option()) &&
                 (!orig_nodes[chain.back()].operator_->HasAsyncPart() ||
                  orig_nodes[cur.first].operator_->SupportsAsyncScheduling()))));
      };
      auto commit_chain = [&]() {
        if (chain.size() > 0) {
          CAFFE_ENFORCE(
              chains.insert({chain.front(), chain}).second,
              "Chain ",
              chain.front(),
              " was already added.");
          VLOG(2) << "Added chain: " << chain.front() << "with elements";
          for (auto ch : chain) {
            VLOG(2) << ch << ", ";
          }
          chain.clear();
        }
      };
      auto depth_traverse = [&]() {
        while (cur.second != nodes[cur.first].children_.end() &&
               seen_nodes.find(*cur.second) != seen_nodes.end()) {
          cur.second++;
        }

        if (cur.second != nodes[cur.first].children_.end()) {
          auto next = make_pair(*cur.second, nodes[*cur.second].children_.begin());
          depth_stack.push(cur);
          depth_stack.push(next);
        }
      };
      for (int root_index : initial_frontier) {
        depth_stack.push(
            make_pair(root_index, nodes[root_index].children_.begin()));
        while (depth_stack.size() > 0) {
          cur = depth_stack.top();
          depth_stack.pop();
          if (seen_nodes.find(cur.first) == seen_nodes.end()) {
            seen_nodes.insert(cur.first);
            // Has one child, can be candidate for chain or can be added to the
            // previous chain.
            if (nodes[cur.first].children_.size() == 1) {
              if (check_current_for_chaining()) {
                // Add oneself to the current chain.
                VLOG(1) << "Adding to existing chain" << cur.first;
                chain.push_back(cur.first);
                int index = *nodes[cur.first].children_.begin();
                depth_stack.push(make_pair(index, nodes[index].children_.begin()));
              } else {
                // Can't belong to the previous chain, commit previous chain and
                // start a new one.
                commit_chain();
                chain.push_back(cur.first);
                int index = *nodes[cur.first].children_.begin();
                depth_stack.push(make_pair(index, nodes[index].children_.begin()));
              }
            } else if (
                nodes[cur.first].children_.size() == 0 &&
                check_current_for_chaining()) {
              // Add current node to the current chain and commit.
              chain.push_back(cur.first);
              commit_chain();
            } else {
              // Node has more than one child.
              commit_chain();
              // Add current node as an independent chain since it won't be a part
              // of a bigger chain.
              chain.push_back(cur.first);
              commit_chain();
              depth_traverse();
            }
          } else {
            // This node has been seen before, we will only traverse its children.
            // Commit any pending chains and continue traversing.
            commit_chain();
            depth_traverse();
          }
        } // End while

        // Check if this if is even needed.
        commit_chain();
      }
      CAFFE_ENFORCE(
          seen_nodes.size() == nodes.size(),
          "Haven't seen all the nodes, expected number of nodes ",
          nodes.size(),
          ", but seen only ",
          seen_nodes.size(),
          ".");

      updateOperatorNodes(orig_nodes, chains);
      return chains;
    */
}
