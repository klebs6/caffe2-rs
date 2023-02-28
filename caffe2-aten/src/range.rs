crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/Range.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/Range.cpp]

pub struct Range {
    begin: i64,
    end:   i64,
}

impl Div<i64> for Range {

    type Output = Range;
    
    #[inline] fn div(self, other: i64) -> Self::Output {
        todo!();
        /*
            return Range(begin / divisor, end / divisor);
        */
    }
}

impl fmt::Display for Range {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            out << "Range[" << range.begin << ", " << range.end << "]";
      return out;
        */
    }
}

impl Range {
    
    pub fn new(
        begin: i64,
        end:   i64) -> Self {
    
        todo!();
        /*
        : begin(begin),
        : end(end),
        */
    }
    
    pub fn size(&self) -> i64 {
        
        todo!();
        /*
            return end - begin;
        */
    }
}
