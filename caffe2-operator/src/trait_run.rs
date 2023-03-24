crate::ix!();

pub trait RunOnDevice {

    fn run_on_device(&mut self) -> bool;
}

pub trait Run {

    #[inline] fn run(&mut self, stream_id: i32) -> bool {
        
        todo!();
        /*
            CAFFE_NOT_IMPLEMENTED;
        */
    }
}
