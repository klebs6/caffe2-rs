crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/TH/THVector.h]

/**
  | We are going to use dynamic dispatch,
  | and want only to generate declarations
  | of the vector functions
  |
  */
#[macro_export] macro_rules! th_vector {
    ($NAME:ident) => {
        /*
                TH_CONCAT_4(TH,Real,Vector_,NAME)
        */
    }
}
