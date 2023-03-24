crate::ix!();

pub trait CheckIsInputOutputAlias {

    /**
      | Check whether output j is an alias of
      | input i by comparing Blob pointers, note
      | this does not check if the two Blobs
      | points to the same Tensor, or if the
      | Tensor pointers point to the same
      | TensorImpl, or if the Storages alias
      */
    #[inline] fn is_input_output_alias(
        &mut self, 
        i: i32,
        j: i32) -> bool 
    {
        todo!();
        /*
            CAFFE_ENFORCE(
            isLegacyOperator(),
            "IsInputOutputAlias(i, j) not (yet) supported for operators exported to c10.");
        return inputs_.at(i) == outputs_.at(j);
        */
    }
}
