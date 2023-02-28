crate::ix!();

use crate::{
    BlobSerializerBase,
    OperatorStorage,
    SerializationAcceptor,
    TypeMeta,
    Blob,
    TensorTypes,
    BlobProto,
    Tensor,
    CPUContext
};

pub type IndexKeyTypes = TensorTypes<(i32, i64, String)>;

///-----------------------------------------
pub struct IndexBase {

    max_elements:  i64,
    meta:          TypeMeta,

    next_id:       i64, //{1}; // guarded by dictMutex_
    frozen_:       AtomicBool, //{false};
    dict_mutex:    parking_lot::RawMutex,
}

impl IndexBase {
    
    pub fn new(max_elements: i64, ty: TypeMeta) -> Self {
        todo!();
        /*
            : maxElements_{maxElements}, meta_(type), frozen_{false}
        */
    }
    
    #[inline] pub fn freeze(&mut self)  {
        
        todo!();
        /*
            frozen_ = true;
        */
    }
    
    #[inline] pub fn is_frozen(&self) -> bool {
        
        todo!();
        /*
            return frozen_;
        */
    }
    
    #[inline] pub fn max_elements(&self) -> i64 {
        
        todo!();
        /*
            return maxElements_;
        */
    }
    
    #[inline] pub fn typemeta(&self) -> TypeMeta {
        
        todo!();
        /*
            return meta_;
        */
    }
    
    #[inline] pub fn size(&mut self) -> i64 {
        
        todo!();
        /*
            std::lock_guard<std::mutex> guard(dictMutex_);
        return nextId_;
        */
    }
}

///-----------------------------------------
pub struct Index<T> {
    base: IndexBase,
    dict: HashMap<T,i64>,
}

impl<T> Index<T> {
    
    pub fn new(max_elements: i64) -> Self {
        todo!();
        /*
            : IndexBase(maxElements, TypeMeta::Make<T>())
        */
    }
    
    #[inline] pub fn get(
        &mut self, 
        keys:     *const T,
        values:   *mut i64,
        num_keys: usize)  
    {
        
        todo!();
        /*
            if (frozen_) {
          FrozenGet(keys, values, numKeys);
          return;
        }
        std::lock_guard<std::mutex> lock(dictMutex_);
        for (int i = 0; i < numKeys; ++i) {
          auto it = dict_.find(keys[i]);
          if (it != dict_.end()) {
            values[i] = it->second;
          } else if (nextId_ < maxElements_) {
            auto newValue = nextId_++;
            dict_.insert({keys[i], newValue});
            values[i] = newValue;
          } else {
            CAFFE_THROW("Dict max size reached");
          }
        }
        */
    }
    
    #[inline] pub fn load(
        &mut self, 
        keys: *const T,
        num_keys: usize) -> bool 
    {
        todo!();
        /*
            CAFFE_ENFORCE(
            numKeys <= maxElements_,
            "Cannot load index: Tensor is larger than max_elements.");
        decltype(dict_) dict;
        for (auto i = 0U; i < numKeys; ++i) {
          CAFFE_ENFORCE(
              dict.insert({keys[i], i + 1}).second,
              "Repeated elements found: cannot load into dictionary.");
        }
        // assume no `get` is inflight while this happens
        {
          std::lock_guard<std::mutex> lock(dictMutex_);
          // let the old dict get destructed outside of the lock
          dict_.swap(dict);
          nextId_ = numKeys + 1;
        }
        return true;
        */
    }
    
    #[inline] pub fn store(&mut self, out: *mut Tensor) -> bool {
        
        todo!();
        /*
            std::lock_guard<std::mutex> lock(dictMutex_);
        out->Resize(nextId_ - 1);
        auto outData = out->template mutable_data<T>();
        for (const auto& entry : dict_) {
          outData[entry.second - 1] = entry.first;
        }
        return true;
        */
    }
    
    #[inline] pub fn frozen_get(
        &mut self, 
        keys: *const T,
        values: *mut i64,
        num_keys: usize)
    {
        todo!();
        /*
            for (auto i = 0U; i < numKeys; ++i) {
          auto it = dict_.find(keys[i]);
          values[i] = it != dict_.end() ? it->second : 0;
        }
        */
    }
}

/**
  | Creates a dictionary that maps T keys
  | to consecutive integers from 1 to max_elements.
  | Zero is reserved for unknown keys.
  | 
  | TODO(azzolini): support sizes larger
  | than int32
  |
  */
pub struct IndexCreateOp<T> {
    storage: OperatorStorage,
    context: CPUContext,

    max_elements: i64,

    phantom: PhantomData<T>,
}

num_inputs!{IndexCreate, 0}

num_outputs!{IndexCreate, 1}

outputs!{IndexCreate, 
    0 => ("handle", "Pointer to an Index instance.")
}

args!{IndexCreate, 
    0 => ("max_elements", "Max number of elements, including the zero entry.")
}

scalar_type!{IndexCreate, TensorProto_DataType_UNDEFINED}


impl<T> IndexCreateOp<T> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...),
            maxElements_(OperatorStorage::GetSingleArgument<int>( "max_elements", int::max))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            *OperatorStorage::Output<std::unique_ptr<IndexBase>>(0) =
            std::unique_ptr<IndexBase>(new Index<T>(maxElements_));
        return true;
        */
    }
}

/**
  | Given an index handle and a tensor of
  | keys, return an Int tensor of same shape
  | containing the indices for each of the
  | keys. If the index is frozen, unknown
  | entries are given index 0. Otherwise,
  | new entries are added into the index.
  | 
  | If an insert is necessary but max_elements
  | has been reached, fail.
  |
  */
pub struct IndexGetOp {
    storage: OperatorStorage,
    context: CPUContext, 
}

num_inputs!{IndexGet, 2}

num_outputs!{IndexGet, 1}

inputs!{IndexGet, 
    0 => ("handle", "Pointer to an Index instance."),
    1 => ("keys", "Tensor of keys to be looked up.")
}

outputs!{IndexGet, 
    0 => ("indices", "Indices for each of the keys.")
}

scalar_type!{IndexGet, TensorProto::INT64}

impl IndexGetOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<IndexKeyTypes>::call(this, Input(1));
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            auto& base = OperatorStorage::Input<std::unique_ptr<IndexBase>>(0);
            auto* dict = dynamic_cast_if_rtti<Index<T>*>(base.get());
            CAFFE_ENFORCE(dict, "Wrong dictionary type given input keys.");
            const auto& keys = Input(1);

            auto* values = Output(0, keys.sizes(), at::dtype<int64_tValue>());
            dict->Get(
                keys.data<T>(),
                values->template mutable_data<int64_tValue>(),
                keys.numel());
            return true;
        */
    }
}

/**
  | Loads the index from the given 1-D tensor.
  | Elements in the tensor will be given
  | consecutive indexes starting at 1.
  | Fails if tensor contains repeated elements.
  |
  */
pub struct IndexLoadOp {
    storage: OperatorStorage,
    context: CPUContext,
    skip_first_entry: bool,
}

num_inputs!{IndexLoad, 2}

num_outputs!{IndexLoad, 1}

inputs!{IndexLoad, 
    0 => ("handle", "Pointer to an Index instance."),
    1 => ("items", "1-D tensor with elements starting with index 1.")
}

outputs!{IndexLoad, 
    0 => ("handle", "The input handle.")
}

args!{IndexLoad, 
    0 => ("skip_first_entry", "If set, skips the first entry of the tensor. 
        This allows to load tensors that are aligned with an embedding, 
        where the first entry corresponds to the default 0 index entry.")
}

scalar_type!{IndexLoad, TensorProto_DataType_UNDEFINED}

enforce_inplace!{IndexLoad, vec![(0, 0)]}

impl IndexLoadOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...),
            skipFirstEntry_( OperatorStorage::GetSingleArgument<int>("skip_first_entry", 0))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<IndexKeyTypes>::call(this, Input(1));
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            auto& base = OperatorStorage::Input<std::unique_ptr<IndexBase>>(0);
            auto* dict = dynamic_cast_if_rtti<Index<T>*>(base.get());
            CAFFE_ENFORCE(dict, "Wrong dictionary type given input keys.");
            const auto& keys = Input(1);
            const auto* keys_data = keys.data<T>();
            auto keys_size = keys.numel();
            if (skipFirstEntry_) {
              CAFFE_ENFORCE(keys.numel() > 0);
              ++keys_data;
              --keys_size;
            }
            return dict->Load(keys_data, keys_size);
        */
    }
}

/**
  | Stores the keys of this index in a 1-D
  | tensor. Since element 0 is reserved
  | for unknowns, the first element of the
  | output tensor will be element of index
  | 1.
  |
  */
pub struct IndexStoreOp {
    storage: OperatorStorage,
    context: CPUContext,
}

num_inputs!{IndexStore, 1}

num_outputs!{IndexStore, 1}

inputs!{IndexStore, 
    0 => ("handle", "Pointer to an Index instance.")
}

outputs!{IndexStore, 
    0 => ("items", "1-D tensor with elements starting with index 1.")
}

impl IndexStoreOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& base = OperatorStorage::Input<std::unique_ptr<IndexBase>>(0);
        return DispatchHelper<IndexKeyTypes>::call(this, base->Type());
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            auto& base = OperatorStorage::Input<std::unique_ptr<IndexBase>>(0);
            auto* dict = dynamic_cast_if_rtti<Index<T>*>(base.get());
            CAFFE_ENFORCE(dict);
            return dict->Store(Output(0));
        */
    }
}

/**
  | Freezes the given index, disallowing
  | creation of new index entries.
  | 
  | Should not be called concurrently with
  | IndexGet.
  |
  */
pub struct IndexFreezeOp {
    storage: OperatorStorage,
    context: CPUContext,
}

num_inputs!{IndexFreeze, 1}

num_outputs!{IndexFreeze, 1}

inputs!{IndexFreeze, 
    0 => ("handle", "Pointer to an Index instance.")
}

outputs!{IndexFreeze, 
    0 => ("handle", "The input handle.")
}

scalar_type!{IndexFreeze, TensorProto_DataType_UNDEFINED}

enforce_inplace!{IndexFreeze, vec![(0, 0)]}

impl IndexFreezeOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& base = OperatorStorage::Input<std::unique_ptr<IndexBase>>(0);
        base->Freeze();
        return true;
        */
    }
}

/**
  | Returns the number of entries currently
  | present in the index.
  |
  */
pub struct IndexSizeOp {
    storage: OperatorStorage,
    context: CPUContext,
}

num_inputs!{IndexSize, 1}

num_outputs!{IndexSize, 1}

inputs!{IndexSize, 
    0 => ("handle", "Pointer to an Index instance.")
}

outputs!{IndexSize, 
    0 => ("items", "Scalar int64 tensor with number of entries.")
}

impl IndexSizeOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& base = OperatorStorage::Input<std::unique_ptr<IndexBase>>(0);

        auto* out = Output(0, std::vector<int64_t>{}, at::dtype<int64_tValue>());
        *out->template mutable_data<int64_tValue>() = base->Size();
        return true;
        */
    }
}

register_cpu_operator!{IntIndexCreate, IndexCreateOp<int32_t>}
register_cpu_operator!{LongIndexCreate, IndexCreateOp<int64_t>}
register_cpu_operator!{StringIndexCreate, IndexCreateOp<std::string>}

register_cpu_operator!{IndexGet, IndexGetOp}
register_cpu_operator!{IndexLoad, IndexLoadOp}
register_cpu_operator!{IndexStore, IndexStoreOp}
register_cpu_operator!{IndexFreeze, IndexFreezeOp}
register_cpu_operator!{IndexSize, IndexSizeOp}

no_gradient!{IndexGetOp}
no_gradient!{IntIndexCreate}
no_gradient!{LongIndexCreate}
no_gradient!{StringIndexCreate}

should_not_do_gradient!{IndexFreeze}
should_not_do_gradient!{IndexLoad}
should_not_do_gradient!{IndexStore}
should_not_do_gradient!{IndexSize}

pub struct IndexSerializer {
    base: dyn BlobSerializerBase,
}

impl IndexSerializer {
    
    #[inline] pub fn serialize(
        &mut self, 
        pointer:   *const c_void,
        type_meta: TypeMeta,
        name:      &String,
        acceptor:  SerializationAcceptor)  
    {
        
        todo!();
        /*
            CAFFE_ENFORCE(typeMeta.Match<std::unique_ptr<IndexBase>>());
        const auto& base = *static_cast<const std::unique_ptr<IndexBase>*>(pointer);
        Blob tensor_blob;
        auto* tensor_out = BlobGetMutableTensor(&tensor_blob, CPU);

        if (base->Type().Match<std::string>()) {
          doStore<std::string>(base, tensor_out);
        } else if (base->Type().Match<int32_t>()) {
          doStore<int32_t>(base, tensor_out);
        } else if (base->Type().Match<int64_t>()) {
          doStore<int64_t>(base, tensor_out);
        } else {
          CAFFE_THROW("Index of this type can't be serialized.");
        }

        CAFFE_ENFORCE(
            tensor_out->numel() <= int32_t::max,
            "Index too large to be serialized.");
        BlobProto blob_proto;
        TensorSerializer ser;
        ser.Serialize(
            *tensor_out, name, blob_proto.mutable_tensor(), 0, tensor_out->numel());
        blob_proto.set_name(name);
        blob_proto.set_type("std::unique_ptr<caffe2::IndexBase>");

        std::ostringstream os;
        os << base->maxElements() << " " << base->isFrozen();
        blob_proto.set_content(os.str());

        acceptor(name, SerializeBlobProtoAsString_EnforceCheck(blob_proto));
        */
    }

    #[inline] pub fn do_store<T>(
        &mut self,
        base: &Box<IndexBase>,
        tensor_out: *mut Tensor)
    {
        todo!();
        /*
            auto* dict = dynamic_cast_if_rtti<Index<T>*>(base.get());
            CAFFE_ENFORCE(dict, "Wrong dictionary type.");
            dict->Store(tensor_out);
        */
    }
}

pub struct IndexDeserializer {
    base: dyn BlobDeserializerBase,
}

impl IndexDeserializer {
    
    #[inline] pub fn deserialize(
        &mut self, 
        proto: &BlobProto,
        blob: *mut Blob)  
    {
        todo!();
        /*
            TensorDeserializer deser;
        Blob tensor_blob;
        deser.Deserialize(proto, &tensor_blob);

        std::istringstream is(proto.content());
        int64_t maxElements{int64_t::max};
        bool isFrozen{false};
        is >> maxElements >> isFrozen;

        auto& tensor_in = tensor_blob.template Get<Tensor>();
        auto* base = blob->template GetMutable<std::unique_ptr<IndexBase>>();

        if (tensor_in.IsType<std::string>()) {
          doLoad<std::string>(base, maxElements, tensor_in);
        } else if (tensor_in.IsType<int32_t>()) {
          doLoad<int32_t>(base, maxElements, tensor_in);
        } else if (tensor_in.IsType<int64_t>()) {
          doLoad<int64_t>(base, maxElements, tensor_in);
        } else {
          CAFFE_THROW("Index of this type cannot be deserialized.");
        }

        if (isFrozen) {
          (*base)->Freeze();
        }
        */
    }

    #[inline] pub fn do_load<T>(
        &mut self,
        base: *mut Box<IndexBase>,
        max_elements: i64,
        tensor_in: &Tensor) {
        todo!();
        /*
            base->reset(new Index<T>(maxElements));
            auto* dict = dynamic_cast_if_rtti<Index<T>*>(base->get());
            dict->Load(tensor_in.data<T>(), tensor_in.numel());
        */
    }
}

caffe_known_type!{Box<IndexBase>}

register_blob_serializer!{ 
    /*
    (TypeMeta::Id<std::unique_ptr<caffe2::IndexBase>>()), 
    IndexSerializer
    */
}

register_blob_deserializer!{ 
    /*
    std::unique_ptr<caffe2::IndexBase>, 
    IndexDeserializer
    */
}
