crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/TensorIteratorInternal.h]

pub struct DimCounter<'a> {
    shape:  &'a [i32],
    range:  Range<usize>,
    values: SmallBuffer<i64,4>,
    offset: i64,
}

impl<'a> DimCounter<'a> {
    
    pub fn new(
        shape: &[i32],
        range: Range<usize>

    ) -> Self {
    
        todo!();
        /*
        
        */
    }
    
    pub fn increment(&mut self, step: &[i64; 2])  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn is_done(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn max_2d_step(&self) -> [i64; 2] {
        
        todo!();
        /*
        
        */
    }
}

#[inline] pub fn get_data_ptrs(
        ptrs:    *mut *mut u8,
        base:    &[*mut u8],
        strides: &[i32],
        counter: &[i32])  {
    
    todo!();
        /*
            const i64 ntensors = base.size();
      const i64 ndim = counter.size();
      copy(base.begin(), base.end(), ptrs);
      for (i64 dim = 0; dim < ndim; ++dim) {
        i64 value = counter[dim];
        for (i64 arg = 0; arg < ntensors; ++arg) {
          ptrs[arg] += value * strides[dim * ntensors + arg];
        }
      }
        */
}

#[inline] pub fn serial_for_each(
    shape:     &[i32],
    strides:   &[i32],
    base_ptrs: *mut *mut u8,
    ntensors:  usize,
    loop_:     tensor_iterator_base::Loop2d,
    range:     Range<usize>

) {
    
    todo!();
        /*
            const auto ndim = shape.size();
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          strides.size() == ntensors * max(usize{2}, ndim));

      if (ndim <= 1) {
        if (range.begin == 0) {
          loop(base_ptrs, strides.data(), range.size(), 1);
        } else {
          SmallBuffer<char*, 4> ptrs(ntensors);
          get_data_ptrs(ptrs.data(), {base_ptrs, ntensors}, strides, {range.begin});
          loop(ptrs.data(), strides.data(), range.size(), 1);
        }
      } else {
        SmallBuffer<char*, 4> ptrs(ntensors);
        auto counter = DimCounter(shape, range);
        while (!counter.is_done()) {
          get_data_ptrs(ptrs.data(), {base_ptrs, ntensors}, strides, counter.values);
          auto step = counter.max_2d_step();
          loop(ptrs.data(), strides.data(), step[0], step[1]);
          counter.increment(step);
        }
      }
        */
}
