crate::ix!();

pub trait GetInputBlob {

    #[inline] fn input_blob(&mut self, idx: i32) -> &Blob {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            isLegacyOperator(),
            "InputBlob(idx) not (yet) supported for operators exported to c10.");
        return *inputs_.at(idx);
        */
    }
}

pub trait GetOutputBlob {

    #[inline] fn output_blob(&mut self, idx: i32) -> *mut Blob {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            isLegacyOperator(),
            "OutputBlob(idx) not (yet) supported for operators exported to c10.");
        return outputs_.at(idx);
        */
    }
}
