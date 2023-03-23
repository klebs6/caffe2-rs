crate::ix!();

/**
  | Graph representation of an operator.
  | 
  | Default constructor for resize
  |
  */
#[derive(Default)]
pub struct Node {

    /**
      | The OperatorDef which this node represents.
      |
      */
    op: OperatorDef,

    /**
      | Keeps track of if an operator has been
      | deleted through a transformation.
      |
      */
    active: bool, //true

    /**
      | Stores a pair (idx, blob_list),
      |
      |  idx = index of the child
      |
      |  blob_list = a list of strings, containing
      |  the blobs that connect the nodes
      */
    parents:  HashMap<i32,Vec<String>>,
    children: HashMap<i32,Vec<String>>,
}

impl Node {
    
    /// Alternate constructor
    pub fn new(
        op:       &OperatorDef,
        active:   bool,
        parents:  HashMap<i32,Vec<String>>,
        children: HashMap<i32,Vec<String>>) -> Self 
    {
        todo!();
        /*
            : op(op), active(active), parents(parents), children(children)
        */
    }
}
