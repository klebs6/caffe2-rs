crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/blob.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/blob.cpp]

/**
  | -----------
  | @brief
  | 
  | Blob is a general container that hosts
  | a typed pointer.
  | 
  | A Blob hosts a pointer as well as its type,
  | and takes charge of deleting it properly
  | when the blob is deallocated or re-allocated
  | with a new type. A blob could contain
  | anything, although the most common
  | case is to contain a Tensor.
  |
  */
#[no_copy]
pub struct Blob {
    base:          IntrusivePtrTarget,
    meta:          TypeMeta,
    pointer:       *mut c_void,
    has_ownership: bool,
}

impl Default for Blob {
    
    /**
      | Initializes an empty Blob.
      |
      */
    fn default() -> Self {
        todo!();
        /*
        : meta(),
        : pointer(nullptr),
        : has_ownership(false),

        
        */
    }
}

impl Drop for Blob {

    fn drop(&mut self) {
        todo!();
        /*
            Reset();
        */
    }
}

impl Blob {
    
    pub fn new(other: Blob) -> Self {
    
        todo!();
        /*
        : blob(),

            swap(other);
        */
    }
    
    pub fn assign_from(&mut self, other: Blob) -> &mut Blob {
        
        todo!();
        /*
            Blob(std::move(other)).swap(*this);
        return *this;
        */
    }

    /**
      | Checks if the content stored in the blob
      | is of type T.
      |
      */
    pub fn is_type<T>(&self) -> bool {
    
        todo!();
        /*
            return meta_.Match<T>();
        */
    }

    /**
      | Returns the meta info of the blob.
      |
      */
    pub fn meta(&self) -> TypeMeta {
        
        todo!();
        /*
            return meta_;
        */
    }

    /**
      | Returns a printable typename of the
      | blob.
      |
      */
    pub fn type_name(&self) -> &str {
        
        todo!();
        /*
            return meta_.name();
        */
    }

    /**
      | -----------
      | @brief
      | 
      | Gets the const reference of the stored
      | object. The code checks if the stored
      | object is of the desired type.
      | 
      | TODO(jerryzh): add a Get(DeviceType)
      | function?
      |
      */
    pub fn get<T>(&self) -> &T {
    
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(
            IsType<T>(),
            "wrong type for the Blob instance. Blob contains ",
            meta_.name(),
            " while caller expects ",
            TypeMeta::TypeName<T>());
        // TODO: after we add Get<Tensor>(DeviceType)
        // and changed all the callsites, we can add
        // a static assert here to enforce T != Tensor
        return *static_cast<const T*>(pointer_);
        */
    }
    
    pub fn get_raw(&self)  {
        
        todo!();
        /*
            return pointer_;
        */
    }
    
    pub fn get_raw_mut(&mut self)  {
        
        todo!();
        /*
            return pointer_;
        */
    }

    /**
      | -----------
      | @brief
      | 
      | Gets a mutable pointer to the stored
      | object.
      | 
      | If the current object is not of the right
      | type, a new object is created and the
      | old object is freed. Note that type T
      | should have a default constructor.
      | Otherwise, create the object yourself
      | first, and use
      | 
      | Reset().
      |
      */
    pub fn get_mutable<T>(&mut self) -> *mut T {
    
        todo!();
        /*
            static_assert(
            std::is_default_constructible<T>::value,
            "GetMutable can't be called with non-default-constructible types. "
            "Try using specialized methods");
        if (IsType<T>()) {
          return static_cast<T*>(pointer_);
        } else {
          // TODO Re-enable logging
          // VLOG(1) << "Create new mutable object " << TypeMeta::TypeName<T>();
          return Reset<T>(new T());
        }
        */
    }
    
    pub fn get_mutable_or_null<T>(&mut self) -> *mut T {
    
        todo!();
        /*
            if (IsType<T>()) {
          return static_cast<T*>(pointer_);
        } else {
          return nullptr;
        }
        */
    }

    /**
      | Sets the underlying object to the allocated
      | one. The Blob then takes over the ownership
      | of the passed in pointer. If there is
      | already an object in the Blob, the old
      | object is freed.
      | 
      | This is used when the underlying class
      | T does not have a default ctor, or complex
      | initializations needs to be done outside
      | the blob.
      |
      */
    pub fn reset_to_allocated<T>(&mut self, allocated: *mut T) -> *mut T {
    
        todo!();
        /*
            free_();
        meta_ = TypeMeta::Make<T>();
        pointer_ = static_cast<void*>(allocated);
        has_ownership_ = true;
        return allocated;
        */
    }

    /**
      | Sets the underlying object to the allocated
      | one, but does not take over the ownership
      | of the passed in pointer. If there is
      | already an object in the Blob, the old
      | object is freed.
      | 
      | Unlike Reset, this does not take over
      | the ownership of the pointer and the
      | caller is responsible for making sure
      | that the lifetime of the allocated blob
      | outlasts the lifetime of any access
      | to this blob, until another Reset call
      | is made or the blob is destructed.
      |
      */
    pub fn share_external<T>(&mut self, allocated: *mut T) -> *mut T {
    
        todo!();
        /*
            return static_cast<T*>(ShareExternal(
            static_cast<void*>(allocated),
            TypeMeta::Make<typename std::remove_const<T>::type>()));
        */
    }
    
    pub fn share_external_with_meta(&mut self, 
        allocated: *mut c_void,
        meta:      TypeMeta)  {
        
        todo!();
        /*
            free_();
        meta_ = meta;
        pointer_ = allocated;
        has_ownership_ = false;
        return allocated;
        */
    }

    /**
      | Resets the Blob to an empty one.
      |
      */
    pub fn reset(&mut self)  {
        
        todo!();
        /*
            free_();
        pointer_ = nullptr;
        meta_ = TypeMeta();
        has_ownership_ = false;
        */
    }

    /**
      | -----------
      | @brief
      | 
      | Swaps the underlying storage of two
      | blobs.
      |
      */
    pub fn swap(&mut self, rhs: &mut Blob)  {
        
        todo!();
        /*
            using std::swap;
        swap(meta_, rhs.meta_);
        swap(pointer_, rhs.pointer_);
        swap(has_ownership_, rhs.has_ownership_);
        */
    }
    
    pub fn free(&mut self)  {
        
        todo!();
        /*
            if (has_ownership_ && pointer_ != nullptr) {
          (*meta_.deleteFn())(pointer_);
        }
        */
    }
}

#[inline] pub fn swap(
        lhs: &mut Blob,
        rhs: &mut Blob)  {
    
    todo!();
        /*
            lhs.swap(rhs);
        */
}

impl fmt::Display for Blob {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            return out << "Blob[" << v.TypeName() << "]";
        */
    }
}
