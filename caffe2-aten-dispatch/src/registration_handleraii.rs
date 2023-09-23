crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/dispatch/RegistrationHandleRAII.h]

pub struct RegistrationHandleRAII {
    on_destruction: fn() -> (),
}

impl Drop for RegistrationHandleRAII {

    fn drop(&mut self) {
        todo!();
        /*
            if (onDestruction_) {
          onDestruction_();
        }
        */
    }
}

impl RegistrationHandleRAII {
    
    pub fn new(on_destruction: fn() -> ()) -> Self {
    
        todo!();
        /*


            : onDestruction_(move(onDestruction))
        */
    }
    
    pub fn new(rhs: RegistrationHandleRAII) -> Self {
    
        todo!();
        /*


            : onDestruction_(move(rhs.onDestruction_)) 

        rhs.onDestruction_ = nullptr;
        */
    }
    
    pub fn assign_from(&mut self, rhs: RegistrationHandleRAII) -> &mut RegistrationHandleRAII {
        
        todo!();
        /*
            onDestruction_ = move(rhs.onDestruction_);
        rhs.onDestruction_ = nullptr;
        return *this;
        */
    }
}
