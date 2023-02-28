crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/util/ScopeExit.h]

/**
  | Mostly copied from https://llvm.org/doxygen/ScopeExit_8h_source.html
  |
  */
pub struct ScopeExit<Callable: Fn() -> ()> {

    exit_function: Callable,

    /**
      | False once moved-from or release()d.
      |
      */
    engaged:       bool, // default = true
}

impl<Callable: Fn() -> ()> Drop for ScopeExit<Callable> {

    fn drop(&mut self) {
        todo!();
        /*
            if (Engaged) {
          ExitFunction();
        }
        */
    }
}

impl<Callable: Fn() -> ()> ScopeExit<Callable> {

    /**
      | constructor accepting a forwarding reference
      | can hide the move constructor
      |
      | @lint-ignore CLANGTIDY
      */
    pub fn new_a<Fp>(F: Fp) -> Self {
    
        todo!();
        /*
            : ExitFunction(forward<Fp>(F))
        */
    }
    
    pub fn new_b(rhs: ScopeExit<Callable>) -> Self {
    
        todo!();
        /*
        : ExitFunction(move(Rhs.ExitFunction)), Engaged(Rhs.Engaged) 

        Rhs.release();
        */
    }
    
    pub fn release(&mut self)  {
        
        todo!();
        /*
            Engaged = false;
        */
    }
}

/**
  | Keeps the callable object that is passed in,
  | and execute it at the destruction of the
  | returned object (usually at the scope exit
  | where the returned object is kept).
  |
  | Interface is specified by p0052r2.
  |
  */
pub fn make_scope_exit<Callable: Fn() -> ()>(F: Callable) -> ScopeExit<Callable> {

    todo!();
        /*
            return scope_exit<typename decay<Callable>::type>(
          forward<Callable>(F));
        */
}
