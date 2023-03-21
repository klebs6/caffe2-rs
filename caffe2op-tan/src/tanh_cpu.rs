crate::ix!();

#[cfg(caffe2_use_accelerate)]
impl TanhFunctor<CPUContext> {

    #[inline] pub fn invoke_f32(&self, 
        n:       i32,
        x:       *const f32,
        y:       *mut f32,
        context: *mut CPUContext) -> bool {
        
        todo!();
        /*
            vvtanhf(Y, X, &N);
      return true;
        */
    }
}
