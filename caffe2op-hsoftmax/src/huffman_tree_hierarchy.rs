crate::ix!();

/**
  | HuffmanTreeHierarchy is an operator
  | to generate huffman tree hierarchy
  | given the input labels. It returns the
  | tree as serialized HierarchyProto
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct HuffmanTreeHierarchyOp<T, Context> {

    storage:     OperatorStorage,
    context:     Context,

    num_classes: i32,

    phantom:     PhantomData<T>,
}

num_inputs!{HuffmanTreeHierarchy, 1}

num_outputs!{HuffmanTreeHierarchy, 1}

inputs!{HuffmanTreeHierarchy, 
    0 => ("Labels", "The labels vector")
}

outputs!{HuffmanTreeHierarchy, 
    0 => ("Hierarch", "Huffman coding hierarchy of the labels")
}

args!{HuffmanTreeHierarchy, 
    0 => ("num_classes", "The number of classes used to build the hierarchy.")
}

should_not_do_gradient!{HuffmanTreeHierarchyOp}


impl<T, Context> HuffmanTreeHierarchyOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            num_classes_(this->template GetSingleArgument<int>("num_classes", -1))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& Y = Input(0);

      CAFFE_ENFORCE_EQ(Y.dim(), 1, "Input labels must be a vector.");
      const auto y_data = Y.template data<T>();
      auto treeOutput = Output(0, {1}, at::dtype<string>());
      std::vector<int> labelCounts;
      labelCounts.resize(num_classes_, 0);
      for (int i = 0; i < Y.dim32(0); ++i) {
        // Labels are in range [0, num_classes]
        const int label_index = y_data[i];
        CAFFE_ENFORCE_LT(
            label_index,
            num_classes_,
            "Found an input label ",
            label_index,
            " not in range [",
            0,
            ",",
            num_classes_,
            "]");
        labelCounts[label_index]++;
      }

      std::priority_queue<Node, std::vector<Node>, NodeComparator> nodes;
      std::vector<Node> huffmanTree;
      std::vector<int> labelIndices;
      labelIndices.resize(num_classes_);

      int current_node_index = 0;
      for (int i = 0; i < num_classes_; ++i) {
        Node node(i, labelCounts[i]);
        nodes.push(node);
      }

      // Extract node with minimum count and insert it in the tree array.
      auto get_next_node = [&nodes, &huffmanTree, &labelIndices]() {
        auto node = nodes.top();
        int node_index = huffmanTree.size();
        if (node.label != -1) {
          labelIndices[node.label] = node_index;
        }
        nodes.pop();
        huffmanTree.push_back(node);
        return std::pair<int, Node>(node_index, node);
      };

      // Merge two nodes and insert the results in the queue.
      auto merge_nodes = [&nodes](
          const std::pair<int, Node>& node_l, const std::pair<int, Node>& node_r) {
        Node node(-1, node_l.second.count + node_r.second.count);
        node.left_ch_index = node_l.first;
        node.right_ch_index = node_r.first;
        nodes.push(node);
      };

      // Main loop for buttom up huffman tree construction.
      while (!nodes.empty()) {
        auto lNode = get_next_node();
        if (!nodes.empty()) {
          auto rNode = get_next_node();
          merge_nodes(lNode, rNode);
        }
      }

      auto is_leaf_node = [&huffmanTree](const int node_index) {
        return huffmanTree[node_index].left_ch_index == -1 &&
            huffmanTree[node_index].right_ch_index == -1;
      };

      auto get_node_label = [&huffmanTree](const int node_index) {
        return huffmanTree[node_index].label;
      };

      // Build huffman tree.
      int current_offset = 0;
      std::function<void(int, NodeProto*)> build_tree = [&](
          const int node_index, NodeProto* node) {
        if (is_leaf_node(node_index) || node_index == -1) {
          return;
        }
        const int left_ch_index = huffmanTree[node_index].left_ch_index;
        const int right_ch_index = huffmanTree[node_index].right_ch_index;
        if (left_ch_index != -1) {
          if (is_leaf_node(left_ch_index)) {
            node->add_word_ids(get_node_label(left_ch_index));
          } else {
            auto* ch_node = node->add_children();
            ch_node->set_offset(current_offset);
            current_offset += 2;
            build_tree(left_ch_index, ch_node);
          }
        }
        if (right_ch_index != -1) {
          if (is_leaf_node(right_ch_index)) {
            node->add_word_ids(get_node_label(right_ch_index));
            current_offset++;
          } else {
            auto* ch_node = node->add_children();
            ch_node->set_offset(current_offset);
            current_offset += 2;
            build_tree(right_ch_index, ch_node);
          }
        }
      };

      // The last element inserted in the tree is the root.
      const int rootNodeIndex = huffmanTree.size() - 1;
      NodeProto rootNode;
      rootNode.set_offset(current_offset);
      current_offset += 2;
      build_tree(rootNodeIndex, &rootNode);
      TreeProto treeProto;
      *treeProto.mutable_root_node() = rootNode;

      treeProto.SerializeToString(treeOutput->template mutable_data<string>());
      return true;
        */
    }
}
