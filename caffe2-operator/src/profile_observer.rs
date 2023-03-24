crate::ix!();

/**
  | This observer displays a description
  | of each operator executed in a network.
  | 
  | This includes input and tensors (name,
  | size, type), arguments, and execution
  | time. This can be used to analyze different
  | performance characteristics.
  | 
  | -----------
  | @note
  | 
  | Currently this observer only supports
  | synchronized computation
  |
  */
pub struct ProfileObserver {
    base: OperatorAttachingNetObserver<ProfileOperatorObserver, ProfileObserver>,

    operator_observers: Vec<*const ProfileOperatorObserver>,
}

impl ProfileObserver {
    
    pub fn new(subject: *mut NetBase) -> Self {
        todo!();
        /*
            : OperatorAttachingNetObserver<ProfileOperatorObserver, ProfileObserver>(
                subject,
                this)
        */
    }
}
