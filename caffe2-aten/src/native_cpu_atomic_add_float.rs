crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/AtomicAddFloat.h]

#[inline] pub fn cpu_atomic_add_float(
        dst:    *mut f32,
        fvalue: f32)  {
    
    todo!();
        /*
            typedef union {
        unsigned intV;
        float floatV;
      } uf32_t;

      uf32_t new_value, old_value;
      atomic<unsigned>* dst_intV = (atomic<unsigned>*)(dst);

      old_value.floatV = *dst;
      new_value.floatV = old_value.floatV + fvalue;

      unsigned* old_intV = (unsigned*)(&old_value.intV);
      while (!atomic_compare_exchange_strong(dst_intV, old_intV, new_value.intV)) {
        _mm_pause();
        old_value.floatV = *dst;
        new_value.floatV = old_value.floatV + fvalue;
      }
        */
}
