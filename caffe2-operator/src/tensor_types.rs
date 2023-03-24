crate::ix!();

pub struct FixedValues<Values> {
    values: PhantomData<Values>,
}

pub struct TensorTypes<Types>  {
    types: PhantomData<Types>,
}

/**
  | Special tag that can be listed in TensorTypes
  | to denote that a special implementation
  | in 'RunWithOtherType' needs to be called
  | instead of failing
  | 
  | Obviously this needs to be the last item
  | in lists, e.g.
  | 
  | TensorTypes<float, double, GenericTensorImplementation>
  |
  */
pub struct GenericTensorImplementation {}

/// Same as TensorTypes but call DoRunWithType2
pub struct TensorTypes2<Types> {
    types: PhantomData<Types>,
}


