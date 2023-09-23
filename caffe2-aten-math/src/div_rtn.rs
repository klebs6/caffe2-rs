crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/div_rtn.h]

/**
  | Integer division rounding to -Infinity
  |
  */
#[inline] pub fn div_rtn<T>(x: T, y: T) -> T {

    todo!();
        /*
            int q = x/y;
        int r = x%y;
        if ((r!=0) && ((r<0) != (y<0))) --q;
        return q;
        */
}
