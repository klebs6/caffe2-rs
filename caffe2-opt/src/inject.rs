crate::ix!();

#[inline] pub fn inject_data_edge_indicators(net: *mut NetDef)  {
    
    todo!();
    /*
        for (const auto& input : net->external_input()) {
        caffe2::OperatorDef op;
        op.set_type("Declare");
        op.add_output(input);
        pushOpToFront(op, net);
      }
      for (const auto& output : net->external_output()) {
        caffe2::OperatorDef op;
        op.set_type("Export");
        op.add_input(output);
        *net->add_op() = std::move(op);
      }
      net->clear_external_input();
      net->clear_external_output();
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
