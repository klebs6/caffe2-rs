crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/THC/THCTensorMathMagma.h]

#[cfg(USE_MAGMA)]
#[inline] pub fn th_magma_malloc_pinned<T>(n: usize) -> *mut T {

    todo!();
        /*
            void* ptr;
      if (MAGMA_SUCCESS != magma_malloc_pinned(&ptr, n * sizeof(T)))
        THError("$ Torch: not enough memory: you tried to allocate %dGB. Buy new RAM!", n/268435456);
      return reinterpret_cast<T*>(ptr);
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/THC/THCTensorMathMagma.cpp]

#[cfg(not(DIVUP))]
#[macro_export] macro_rules! divup {
    ($x:ident, $y:ident) => {
        /*
                (((x) + (y) - 1) / (y))
        */
    }
}

#[macro_export] macro_rules! no_magma {
    ($name:ident) => {
        /*
                "No CUDA implementation of '" #name "'. Install MAGMA and rebuild cutorch (http://icl.cs.utk.edu/magma/)"
        */
    }
}

pub fn thc_magma_init()  {
    
    todo!();
        /*
            #ifdef USE_MAGMA
      magma_init();
    #endif
        */
}

pub struct Initializer {

}

impl Default for Initializer {
    
    fn default() -> Self {
        todo!();
        /*


            ::THCMagma_init = _THCMagma_init;
      }{
        */
    }
}
