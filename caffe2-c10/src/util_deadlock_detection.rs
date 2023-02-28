/*!
  | This file provides some simple utilities for
  | detecting common deadlocks in PyTorch.  For
  | now, we focus exclusively on detecting Python
  | GIL deadlocks, as the GIL is a wide ranging
  | lock that is taken out in many situations. The
  | basic strategy is before performing an
  | operation that may block, you can use
  | TORCH_ASSERT_NO_GIL_WITHOUT_PYTHON_DEP() to
  | assert that the GIL is not held.  This macro
  | is to be used in contexts where no static
  | dependency on Python is available (we will
  | handle indirecting a virtual call for you).
  |
  | If the GIL is held by a torchdeploy
  | interpreter, we always report false.
  |
  | If you are in a context where Python bindings
  | are available, it's better to directly assert
  | on PyGILState_Check (as it avoids a vcall and
  | also works correctly with torchdeploy.)
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/util/DeadlockDetection.h]

macro_rules! TORCH_ASSERT_NO_GIL_WITHOUT_PYTHON_DEP {
    () => {

        /*
        TORCH_INTERNAL_ASSERT(                         
            !check_python_gil(),            
            "Holding GIL before a blocking operation!  Please release the GIL before blocking, or see https://github.com/pytorch/pytorch/issues/56297 for how to release the GIL for destructors of objects"
        )
        */
    }
}

/**
 | Returns true if we hold the GIL. If not
 | linked against Python we always return
 | false.
 |
 */
pub trait PythonGILHooksInterface:
CheckPythonGil {}

pub trait CheckPythonGil {

    
    fn check_python_gil(&self) -> bool;
}

/**
  | DO NOT call this registerer from a torch
  | deploy instance! You will clobber other
  | registrations
  |
  */
pub struct PythonGILHooksRegisterer {

}

impl Drop for PythonGILHooksRegisterer {

    fn drop(&mut self) {
        todo!();
        /*
            SetPythonGILHooks(nullptr);
        */
    }
}

impl PythonGILHooksRegisterer {
    
    pub fn new(factory: *mut dyn PythonGILHooksInterface) -> Self {
    
        todo!();
        /*
            SetPythonGILHooks(factory);
        */
    }
}

//-------------------------------------------[.cpp/pytorch/c10/util/DeadlockDetection.cpp]

lazy_static!{
    /*
    PythonGILHooks* python_gil_hooks = nullptr;
    */
}

pub fn check_python_gil() -> bool {
    
    todo!();
        /*
            if (!python_gil_hooks) {
        return false;
      }
      return python_gil_hooks->check_python_gil();
        */
}

pub fn set_python_gil_hooks(hooks: *mut dyn PythonGILHooksInterface)  {
    
    todo!();
        /*
            TORCH_INTERNAL_ASSERT(!hooks || !python_gil_hooks);
      python_gil_hooks = hooks;
        */
}
