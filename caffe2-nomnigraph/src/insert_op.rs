crate::ix!();

#[inline] pub fn insert_op<T, U, Args>(
    g:    &mut NNGraph,
    a:    NodeRef<T,U>,
    b:    NodeRef<T,U>,
    args: Args)  {

    todo!();
    /*
        if (is<NeuralNetData>(a) && is<NeuralNetOperator>(b)) {
        auto newNode = g.createNode(std::make_unique<T>(args));
        auto data = get<NeuralNetData>(a);
        auto newData =
            g.createNode(std::make_unique<NomTensor>(data->getName() + "_"));
        g.createEdge(a, newNode);
        g.createEdge(newNode, newData);
        g.createEdge(newData, b);
        return;
      }
      if (is<NeuralNetOperator>(a) && is<NeuralNetData>(b)) {
        auto newNode = g.createNode(std::make_unique<T>(args));
        auto data = get<NeuralNetData>(b);
        auto newData =
            g.createNode(std::make_unique<NomTensor>(data->getName() + "_"));
        g.createEdge(a, newData);
        g.createEdge(newData, newNode);
        g.createEdge(newNode, b);
        return;
      }

      assert(0 && "insertOp takes (DFG, NomTensor, Op) or (DFG, Op, NomTensor)");
    */
}
