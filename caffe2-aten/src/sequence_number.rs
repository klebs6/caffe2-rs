crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/SequenceNumber.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/SequenceNumber.cpp]

lazy_static!{
    /*
    thread_local u64 sequence_nr_ = 0;
    */
}

/**
  | A simple thread local enumeration, used to link
  | forward and backward pass ops and is used by
  | autograd and observers framework
  |
  */
pub fn sequence_number_peek() -> u64 {
    
    todo!();
        /*
            return sequence_nr_;
        */
}

pub fn sequence_number_get_and_increment() -> u64 {
    
    todo!();
        /*
            return sequence_nr_++;
        */
}
