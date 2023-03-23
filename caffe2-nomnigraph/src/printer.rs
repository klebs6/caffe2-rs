crate::ix!();

#[inline] pub fn bb_printer<T,U>(node: NodeRef<T,U>) 
-> HashMap<String,String> 
{
    todo!();
    /*
        std::map<std::string, std::string> labelMap;
      auto& bb = node->data();
      labelMap["label"] = to_string((unsigned long long)node) + "\\n";
      for (const auto& instr : bb.getInstructions()) {
        assert(isa<nom::repr::NeuralNetOperator>(instr->data()) &&
               "Invalid instruction.");
        auto *op = dyn_cast<nom::repr::NeuralNetOperator>(instr->data().get());
        bool hasOutput = false;
        for (const auto &outEdge : instr->getOutEdges()) {
          auto *output =
              dyn_cast<nom::repr::NeuralNetData>(outEdge->head()->data().get());
          labelMap["label"] += " " + output->getName();
          hasOutput = true;
        }
        if (hasOutput) {
          labelMap["label"] += " = ";
        }
        labelMap["label"] += op->getName();
        for (const auto &inEdge : instr->getInEdges()) {
          auto *arg =
              dyn_cast<nom::repr::NeuralNetData>(inEdge->tail()->data().get());
          labelMap["label"] += " " + arg->getName();
        }
        labelMap["label"] += "\\l";
      }
      labelMap["shape"] = "box";
      return labelMap;
    */
}

#[inline] pub fn cfg_edge_printer<T,U>(edge: EdgeRef<T,U>) 
-> HashMap<String,String> 
{
    todo!();
    /*
        std::map<std::string, std::string> labelMap;
      if (edge->data() == -1) {
        labelMap["label"] = "F";
      } else if (edge->data() == 1) {
        labelMap["label"] = "T";
      }
      return labelMap;
    */
}

#[inline] pub fn nn_printer<T,U>(node: NodeRef<T,U>) 
-> HashMap<String,String> 
{
    
    todo!();
    /*
        std::map<std::string, std::string> labelMap;
      assert(node->data() && "Node doesn't have data, can't render it");
      if (isa<nom::repr::NeuralNetOperator>(node->data())) {
        auto *op = dyn_cast<nom::repr::NeuralNetOperator>(node->data().get());
        labelMap["label"] =
            op->getName() + " (" + to_string((unsigned long long)node) + ")";
        labelMap["shape"] = "box";
      } else if (isa<nom::repr::Data>(node->data())) {
        auto tensor = dyn_cast<nom::repr::NeuralNetData>(node->data().get());
        labelMap["label"] = tensor->getName();
        labelMap["label"] += "_" + to_string(tensor->getVersion()) + " " + to_string((unsigned long long)node);
      }
      return labelMap;
    */
}

pub fn create_test_node(g: &mut NomGraph<TestClass>) 
    -> <NomGraph::<TestClass> as GraphType>::NodeRef 
{
    todo!();
    /*
       return g.createNode(TestClass());
       */
}

pub fn test_node_printer(unused: <NomGraph<TestClass> as GraphType>::NodeRef) 
    -> HashMap<String, String> 
{
    todo!();
    /*
       std::map<std::string, std::string> labelMap;
       labelMap["label"] = "Node";
       return labelMap;
       */
}

pub struct TestClass { }

/** 
 | Our test graph looks like this:
 |           +-------+
 |           | entry |
 |           +-------+
 |             |
 |             |
 |             v
 |           +-------+
 |           |   1   |
 |           +-------+
 |             |
 |             |
 |             v
 | +---+     +-------+
 | | 4 | <-- |   2   |
 | +---+     +-------+
 |   |         |
 |   |         |
 |   |         v
 |   |       +-------+
 |   |       |   3   |
 |   |       +-------+
 |   |         |
 |   |         |
 |   |         v
 |   |       +-------+
 |   +-----> |   6   |
 |           +-------+
 |             |
 |             |
 |             v
 | +---+     +-------+
 | | 5 | --> |   7   |
 | +---+     +-------+
 |             |
 |             |
 |             v
 |           +-------+
 |           | exit  |
 |           +-------+
 |
 | Here is the code used to generate the dot file
 | for it:
 |
 |  auto str = nom::converters::convertToDotString(&graph,
 |    [](nom::Graph<std::string>::NodeRef node) {
 |      std::map<std::string, std::string> labelMap;
 |      labelMap["label"] = node->data();
 |      return labelMap;
 |    });
 */
#[inline] pub fn to_string<T>(value: T) -> String {

    todo!();
    /*
        std::ostringstream os;
        os << value;
        return os.str();
    */
}
