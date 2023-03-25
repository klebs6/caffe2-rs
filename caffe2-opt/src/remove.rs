crate::ix!();

#[inline] pub fn remove_data_edge_indicators(net: *mut NetDef)  {
    
    todo!();
    /*
        google::protobuf::RepeatedPtrField<caffe2::OperatorDef>* op_list(
          net->mutable_op());
      for (auto i = 0; i < net->op_size(); ++i) {
        auto op = net->op(i);
        if (op.type() == "Declare") {
          net->add_external_input(op.output(0));
        } else if (op.type() == "Export") {
          net->add_external_output(op.input(0));
        } else {
          continue;
        }
        // Note that this compensates for modifying the list inplace
        op_list->DeleteSubrange(i--, 1);
      }
    */
}

#[inline] pub fn remove_data_edge_indicators_with_nnmodule<T,U>(nn: *mut NNModule<T,U>)  {
    
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
