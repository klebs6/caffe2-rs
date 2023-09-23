crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/THC/THCGenerateAllTypes.h]
pub const THCTypeIdxByte:   usize = 1;
pub const THCTypeIdxChar:   usize = 2;
pub const THCTypeIdxShort:  usize = 3;
pub const THCTypeIdxInt:    usize = 4;
pub const THCTypeIdxLong:   usize = 5;
pub const THCTypeIdxFloat:  usize = 6;
pub const THCTypeIdxDouble: usize = 7;
pub const THCTypeIdxHalf:   usize = 8;

#[macro_export] macro_rules! thc_type_idx {
    ($T:ident) => {
        /*
                TH_CONCAT_2(THCTypeIdx,T)
        */
    }
}
