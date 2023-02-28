crate::ix!();

use crate::*;

/**
  | A one to one operator that takes an arbitrary
  | number of input and output blobs such
  | that each input blob is inplace with
  | it's matching output blob. It then proceedes
  | to do nothing with each of these operators.
  | This serves two purposes. It can make
  | it appear as if a blob has been written
  | to, as well as can tie together different
  | blobs in a data dependency
  |
  */
pub struct DataCoupleOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

impl<Context> DataCoupleOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // Actually does nothing...
        return true;
        */
    }
}

register_cpu_operator!{DataCouple, DataCoupleOp<CPUContext>}

enforce_one_to_one_inplace!{DataCouple}
