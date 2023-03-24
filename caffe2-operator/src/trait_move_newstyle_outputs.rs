crate::ix!();

pub trait MoveNewstyleOutputs {

    #[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
    #[inline] fn move_newstyle_outputs(&mut self) -> List<Tensor> {
        
        todo!();
        /*
            return std::move(newstyle_outputs_);
        */
    }
}
