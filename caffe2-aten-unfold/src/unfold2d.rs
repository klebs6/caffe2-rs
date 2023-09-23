crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Unfold2d.h]

pub type Unfold2dFn = fn(
        finput:        &mut Tensor,
        input:         &mut Tensor,
        kh:            i64,
        kw:            i64,
        dh:            i64,
        dw:            i64,
        padh:          i64,
        padw:          i64,
        n_input_plane: i64,
        input_height:  i64,
        input_width:   i64,
        output_height: i64,
        output_width:  i64
) -> ();

declare_dispatch!{unfold2d_fn, unfolded2d_copy_stub}
declare_dispatch!{unfold2d_fn, unfolded2d_acc_stub}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Unfold2d.cpp]

define_dispatch!{unfolded2d_copy_stub}
define_dispatch!{unfolded2d_acc_stub}
