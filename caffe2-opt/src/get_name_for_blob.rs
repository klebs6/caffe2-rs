crate::ix!();

/**
  | Helper function for convertToNQLString
  | function.
  |
  | Given a node and a renameMap return the unique
  | name for this node.
  */
#[inline] pub fn get_name_for_blob(node: NNGraph_NodeRef, rename_map: &HashMap<NNGraph_NodeRef,String>) -> String {
    
    todo!();
    /*
        if (renameMap.count(node)) {
        return renameMap.at(node);
      }
      return getNodeName(node);
    */
}

/**
  | Helper function for convertToNQLString
  | function.
  |
  | Given a node and a renameMap return a string
  | representing the node, which looks something
  | like:
  |
  |   %a = Op(%b, %c, %d)
  */
#[inline] pub fn get_nqlstring_for_blob(
    node: NNGraph_NodeRef, 
    rename_map: &HashMap<NNGraph_NodeRef,String>) -> String 
{
    
    todo!();
    /*
        if (!nn::is<Data>(node) || !nn::hasProducer(node)) {
        return "";
      }
      NNGraph_NodeRef defOp = nn::getProducer(node);

      std::string result =
          getNameForBlob(node, renameMap) + " = " + getNodeName(defOp) + "(";
      int i = 0;
      for (auto inputTensor : nn::getInputs(defOp)) {
        if (i) {
          result += ", ";
        }
        result += getNameForBlob(inputTensor, renameMap);
        i++;
      }
      result += ")";
      return result;
    */
}

