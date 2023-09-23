
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/AlignedAllocator.h]

lazy_static!{
    /*
    template <typename T, usize Alignment>
    class AlignedAllocator;

    template <usize Alignment>
    class AlignedAllocator<void, Alignment> {
     
      typedef void* pointer;
      typedef const void* const_pointer;
      typedef void value_type;

      template <class U>
      struct rebind {
        typedef AlignedAllocator<U, Alignment> other;
      };
    };
    */
}

pub struct AlignedAllocator<T,const Alignment: usize> {

}

pub mod aligned_allocator {

    use super::*;

    pub type ValueType      = T;
    pub type Pointer        = *mut T;
    pub type ConstPointer   = *const T;
    pub type Refernce       = Rc<RefCell<T>>;
    pub type ConstReference = Rc<T>;
    pub type SizeType       = usize;
    pub type DifferenceType = libc::ptrdiff_t;

    #[cfg(__cplusplus_GTE_201402L)]
    pub type propagate_on_container_move_assignment = TrueType;

    pub mod rebind {

        use super::*;

        pub type Other<U> = AlignedAllocator<U, Alignment>;
    }
}

impl<T,const Alignment: usize> AlignedAllocator<T,Alignment> {
    
    pub fn new() -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn new<U>(other: &AlignedAllocator<U,Alignment>) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    #[inline] pub fn max_size(&self) -> SizeType {
        
        todo!();
        /*
            return (Sizeype::max - Sizeype(Alignment)) /
            sizeof(T);
        */
    }
    
    #[inline] pub fn address(&self, x: Reference) -> Pointer {
        
        todo!();
        /*
            return addressof(x);
        */
    }
    
    #[inline] pub fn address(&self, x: ConstReference) -> ConstPointer {
        
        todo!();
        /*
            return addressof(x);
        */
    }
    
    #[inline] pub fn allocate(&mut self, 
        n:    SizeType,
        hint: AlignedAllocator<void, Alignment>::const_pointer) -> Pointer {
        let hint: AlignedAllocator<void, Alignment>::const_pointer = hint.unwrap_or(0);

        todo!();
        /*
            #if defined(__ANDROID__)
        void* memory = memalign(Alignment, n * sizeof(T));
        if (memory == 0) {
    #if !defined(__GNUC__) || defined(__EXCEPTIONS)
          throw bad_alloc();
    #endif
        }
    #else
        void* memory = nullptr;
        if (posix_memalign(&memory, Alignment, n * sizeof(T)) != 0) {
    #if !defined(__GNUC__) || defined(__EXCEPTIONS)
          throw bad_alloc();
    #endif
        }
    #endif
        return static_cast<pointer>(memory);
        */
    }
    
    #[inline] pub fn deallocate(&mut self, 
        p: Pointer,
        n: SizeType)  {
        
        todo!();
        /*
            free(static_cast<void*>(p));
        */
    }
    
    
    #[inline] pub fn construct<U, Args>(&mut self, 
        p:    *mut U,
        args: Args)  {
    
        todo!();
        /*
            ::new (static_cast<void*>(p)) U(forward<Args>(args)...);
        */
    }
    
    
    #[inline] pub fn destroy<U>(&mut self, p: *mut U)  {
    
        todo!();
        /*
            p->~U();
        */
    }
}
