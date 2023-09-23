// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/TH/generic/THStorage.h]
lazy_static!{
    /*
    #ifndef TH_GENERIC_FILE
    #define TH_GENERIC_FILE "TH/generic/THStorage.h"
    #else


    /* on pourrait avoir un liste chainee
       qui initialise math, lab structures (or more).
       mouais -- complique.

       Pb: THMapStorage is kind of a class
       THLab_()... comment je m'en sors?

       en template, faudrait que je les instancie toutes!!! oh boy!
       Et comment je sais que c'est pour Cuda? Le type float est le meme dans les <>

       au bout du compte, ca serait sur des pointeurs float/double... etc... = facile.
       primitives??
     */

    // Struct definition is moved to THStorage.hpp (so this file stays C compatible)

    #define THStorage StorageImpl

    // These used to be distinct types; for some measure of backwards compatibility and documentation
    // alias these to the single THStorage type.
    #define THFloatStorage THStorage
    #define THDoubleStorage THStorage
    #define THHalfStorage THStorage
    #define THByteStorage THStorage
    #define THCharStorage THStorage
    #define THShortStorage THStorage
    #define THIntStorage THStorage
    #define THLongStorage THStorage
    #define THBoolStorage THStorage
    #define THBFloat16Storage THStorage
    #define THQUInt8Storage THStorage
    #define THQInt8Storage THStorage
    #define THQInt32Storage THStorage
    #define THQUInt4x2Storage THStorage
    #define THComplexFloatStorage THStorage
    #define THComplexDoubleStorage THStorage

    TH_API scalar_t* THStorage_(data)(const THStorage*);
    TH_API usize THStorage_(elementSize)(void);

    /* slow access -- checks everything */
    TH_API void THStorage_(set)(THStorage*, ptrdiff_t, scalar_t);
    TH_API scalar_t THStorage_(get)(const THStorage*, ptrdiff_t);

    TH_API THStorage* THStorage_(new)(void);
    TH_API THStorage* THStorage_(newWithSize)(ptrdiff_t size);
    TH_API THStorage* THStorage_(newWithSize1)(scalar_t);
    TH_API THStorage* THStorage_(newWithMapping)(const char *filename, ptrdiff_t size, int flags);

    TH_API THStorage* THStorage_(newWithAllocator)(ptrdiff_t size,
                                                   Allocator* allocator);
    TH_API THStorage* THStorage_(newWithDataAndAllocator)(
        DataPtr&& data, ptrdiff_t size, Allocator* allocator);

    /* should not differ with API */
    TH_API void THStorage_(setFlag)(THStorage *storage, const char flag);
    TH_API void THStorage_(clearFlag)(THStorage *storage, const char flag);
    TH_API void THStorage_(retain)(THStorage *storage);
    TH_API void THStorage_(swap)(THStorage *storage1, THStorage *storage2);

    /* might differ with other API (like CUDA) */
    TH_API void THStorage_(free)(THStorage *storage);
    TH_API void THStorage_(resizeBytes)(THStorage* storage, ptrdiff_t size_bytes);
    TH_API void THStorage_(fill)(THStorage *storage, scalar_t value);

    #endif
    //-------------------------------------------[.cpp/pytorch/aten/src/TH/generic/THStorage.cpp]
    #ifndef TH_GENERIC_FILE
    #define TH_GENERIC_FILE "TH/generic/THStorage.cpp"
    #else

    scalar_t* THStorage_(data)(const THStorage *self)
    {
    #if defined(THQUANTIZED)
      return reinterpret_cast<scalar_t*>(self->data<quantized_t>());
    #else
      return self->data<scalar_t>();
    #endif
    }

    usize THStorage_(elementSize)()
    {
      return sizeof(scalar_t);
    }

    THStorage* THStorage_(new)(void)
    {
      return THStorage_new();
    }

    THStorage* THStorage_(newWithSize)(ptrdiff_t size)
    {
      THStorage* storage = make_intrusive<StorageImpl>(
                               StorageImpl::use_byte_Size(),
    #ifdef THQUANTIZED
                               size * sizeof(quantized_t),
    #else
                               size * sizeof(scalar_t),
    #endif
                               getTHDefaultAllocator(),
                               true)
                               .release();
      return storage;
    }

    THStorage* THStorage_(newWithAllocator)(ptrdiff_t size,
                                            Allocator *allocator)
    {
      THStorage* storage = make_intrusive<StorageImpl>(
                               StorageImpl::use_byte_Size(),
    #ifdef THQUANTIZED
                               size * sizeof(quantized_t),
    #else
                               size * sizeof(scalar_t),
    #endif
                               allocator,
                               true)
                               .release();
      return storage;
    }

    THStorage* THStorage_(newWithMapping)(const char *filename, ptrdiff_t size, int flags)
    {
      usize actual_size = -1;
      THStorage* storage =
          make_intrusive<StorageImpl>(
              StorageImpl::use_byte_Size(),
              size * sizeof(scalar_t),
              THMapAllocator::makeDataPtr(
                  filename, flags, size * sizeof(scalar_t), &actual_size),
              /* allocator */ nullptr,
              false)
              .release();

      if (size <= 0) {
        storage->set_nbytes(actual_size);
      }

      return storage;
    }

    THStorage* THStorage_(newWithSize1)(scalar_t data0)
    {
      THStorage *self = THStorage_(newWithSize)(1);
      scalar_t *data = THStorage_(data)(self);
      data[0] = data0;
      return self;
    }

    void THStorage_(retain)(THStorage *storage)
    {
      THStorage_retain(storage);
    }

    void THStorage_(free)(THStorage *storage)
    {
      THStorage_free(storage);
    }

    THStorage* THStorage_(newWithDataAndAllocator)(DataPtr&& data, ptrdiff_t size,
                                                   Allocator* allocator) {
      THStorage* storage = make_intrusive<StorageImpl>(
                               StorageImpl::use_byte_Size(),
    #ifdef THQUANTIZED
                               size * sizeof(quantized_t),
    #else
                               size * sizeof(scalar_t),
    #endif
                               move(data),
                               allocator,
                               allocator != nullptr)
                               .release();
      return storage;
    }

    void THStorage_(resizeBytes)(THStorage* storage, ptrdiff_t size_bytes) {
      return THStorage_resizeBytes(storage, size_bytes);
    }

    void THStorage_(fill)(THStorage *storage, scalar_t value)
    {
      auto type_meta = TypeMeta::Make<scalar_t>();
      usize numel = storage->nbytes() / type_meta.itemsize();
      for (usize i = 0; i < numel; i++)
        THStorage_(data)(storage)[i] = value;
    }

    void THStorage_(set)(THStorage *self, ptrdiff_t idx, scalar_t value)
    {
      auto type_meta = TypeMeta::Make<scalar_t>();
      usize numel = self->nbytes() / type_meta.itemsize();
      THArgCheck((idx >= 0) && (idx < numel), 2, "out of bounds");
      THStorage_(data)(self)[idx] = value;
    }

    scalar_t THStorage_(get)(const THStorage *self, ptrdiff_t idx)
    {
      auto type_meta = TypeMeta::Make<scalar_t>();
      usize numel = self->nbytes() / type_meta.itemsize();
      THArgCheck((idx >= 0) && (idx < numel), 2, "out of bounds");
      return THStorage_(data)(self)[idx];
    }

    void THStorage_(swap)(THStorage *storage1, THStorage *storage2)
    {
      swap(*storage1, *storage2);
    }

    #endif
    */
}

