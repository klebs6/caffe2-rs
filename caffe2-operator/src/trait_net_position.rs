crate::ix!();

pub trait CheckNetPosition {

    #[inline] fn net_position(&self) -> i32 {
        
        todo!();
        /*
            return net_position_;
        */
    }
}

pub trait SetNetPosition {

    #[inline] fn set_net_position(&mut self, idx: i32)  {
        
        todo!();
        /*
            net_position_ = idx;
        */
    }
}
