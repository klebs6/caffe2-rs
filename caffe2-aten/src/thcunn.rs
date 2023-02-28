crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/THCUNN/THCUNN.h]

pub type THCIndexTensor = THCudaLongTensor;

#[macro_export] macro_rules! thc_index_tensor {
    ($NAME:ident) => {
        /*
                THCudaLongTensor_ ## NAME
        */
    }
}

pub type THCIndex = i64;

#[macro_export] macro_rules! thnn_ {
    ($NAME:ident) => {
        /*
                TH_CONCAT_3(THNN_, CReal, NAME)
        */
    }
}
