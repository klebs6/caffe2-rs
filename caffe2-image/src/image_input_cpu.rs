crate::ix!();

impl ImageInputOp<CPUContext> {

    #[inline] pub fn apply_transform_onGPU(
        &mut self, 
        x: &Vec<i64>,
        device: &Device) -> bool 
    {
        todo!();
        /*
            return false;
        */
    }
}
