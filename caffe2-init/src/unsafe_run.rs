crate::ix!();

#[inline] pub fn unsafe_run_caffe_2init_function(
    name:  *const u8,
    pargc: *mut i32,
    pargv: *mut *mut *mut u8) -> bool 
{
    todo!();
    /*
        return internal::Caffe2InitializeRegistry::Registry()->RunNamedFunction(
            name, pargc, pargv);
    */
}
