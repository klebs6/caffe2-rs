crate::ix!();

/**
  | Padding mode similar to numpy.
  |
  */
pub enum PadMode {

    /**
      | pad constant values, with string "constant"
      |
      */
    CONSTANT = 0, 

    /**
      | pads with reflect values, with string
      | "reflect"
      |
      */
    REFLECT = 1,

    /**
      | pads with the edge values, with string
      | "edge"
      |
      */
    EDGE = 2,
}

#[inline] pub fn string_to_pad_mode(mode: &String) -> PadMode {
    
    todo!();
    /*
        if (mode == "constant") {
        return PadMode::CONSTANT;
      } else if (mode == "reflect") {
        return PadMode::REFLECT;
      } else if (mode == "edge") {
        return PadMode::EDGE;
      } else {
        CAFFE_THROW("Unknown padding mode: " + mode);
      }
    */
}
