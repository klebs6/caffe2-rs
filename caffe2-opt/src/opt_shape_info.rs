crate::ix!();

use crate::{
    Blob,
    TensorProto_DataType,
    TensorBoundShapes,
    TensorShape,
    TensorBoundShape,
    TensorBoundShape_DimType
};

#[derive(Default)]
pub struct QShapeInfo {
    axis:    u32,
    offset:  Vec<f32>,
    scale:   Vec<f32>,
}

impl QShapeInfo {
    
    pub fn new(
        o: Option<f32>,
        s: Option<f32>,
        a: Option<u32>) -> Self {

        let o: f32 = o.unwrap_or(0.0);
        let s: f32 = s.unwrap_or(1.0);
        let a: u32 = a.unwrap_or(1);

        todo!();
        /*
            offset.clear();
        scale.clear();
        offset.push_back(o);
        scale.push_back(s);
        axis = a;
        */
    }
}

pub struct ShapeInfo {

    shape:  TensorShape,

    /// quantization related information
    is_quantized:  bool,

    q_info:  QShapeInfo,

    /**
      | type of the shape for every dimension
      | dim_type.size == shape.dims.size
      |
      */
    dim_type:  Vec<TensorBoundShape_DimType>,

    dim_type_is_set:  bool, // default = false

    /**
      | a flag to indicate whether the shape
      | is final and cannot be changed eg: input/output
      | of in-place ops
      |
      */
    shape_is_final:  bool, // default = false
}

impl Default for ShapeInfo {
    fn default() -> Self {
        Self {
            shape:            TensorShape::default(),
            is_quantized:     false,
            q_info:           QShapeInfo::default(),
            dim_type:         vec![],
            dim_type_is_set:  false,//TODO calculate
            shape_is_final:   false,
        }
    }
}

impl ShapeInfo {
    
    #[inline] pub fn set_dim_type(&mut self, dim_types: &Vec<TensorBoundShape_DimType>)  {
        
        todo!();
        /*
            if (shape.dims_size()) {
          CAFFE_ENFORCE_EQ(shape.dims_size(), dim_types.size());
        }
        dim_type = dim_types;
        dim_type_is_set = true;
        */
    }
    
    #[inline] pub fn set_dim_type_form_idx_and_ty(&mut self, 
        idx: i32, 
        ty: TensorBoundShape_DimType)
    {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            dim_type.size() > idx, dim_type.size(), "vs", dim_type.size());
        dim_type[idx] = type;
        dim_type_is_set = true;
        */
    }
    
    #[inline] pub fn dim_type_is_set(&mut self) -> bool {
        
        todo!();
        /*
            return dim_type_is_set;
        */
    }
    
    #[inline] pub fn get_dim_type(&self) -> &Vec<TensorBoundShape_DimType> {
        
        todo!();
        /*
            return dim_type;
        */
    }
    
    #[inline] pub fn get_dim_type_from_idx(&self, idx: i32) -> TensorBoundShape_DimType {
        
        todo!();
        /*
            if (dim_type.size() > idx) {
          return dim_type[idx];
        } else {
          return TensorBoundShape_DimType_UNKNOWN;
        }
        */
    }
    
    #[inline] pub fn get_shape_is_final(&mut self) -> bool {
        
        todo!();
        /*
            return shape_is_final;
        */
    }
    
    #[inline] pub fn set_shape_is_final(&mut self, flag: bool)  {
        
        todo!();
        /*
            shape_is_final = flag;
        */
    }
}

pub type ShapeInfoMap = HashMap<String,ShapeInfo>;

#[inline] pub fn is_number(s: &String) -> bool {
    
    todo!();
    /*
        bool empty = true;
      for (const char c : s) {
        if (std::isalpha(c)) {
          return false;
        }
        if (!std::isspace(c)) {
          empty = false;
        }
      }
      return !empty;
    */
}


#[inline] pub fn to_lower(s: &String) -> String {
    
    todo!();
    /*
        std::string t;
      t.resize(s.size());
      for (size_t i = 0; i < t.size(); i++) {
        t[i] = std::tolower(s[i]);
      }
      return t;
    */
}

#[inline] pub fn to_tensor_proto_data_type(input: &String) -> TensorProto_DataType {
    
    todo!();
    /*
        std::string s = toLower(in);
      if (s == "uint8") {
        return TensorProto_DataType_UINT8;
      } else if (s == "int8") {
        return TensorProto_DataType_INT8;
      } else if (s == "uint16") {
        return TensorProto_DataType_UINT16;
      } else if (s == "int16") {
        return TensorProto_DataType_INT16;
      } else if (s == "int32") {
        return TensorProto_DataType_INT32;
      } else if (s == "int64") {
        return TensorProto_DataType_INT64;
      } else if (s == "float16" || s == "half") {
        return TensorProto_DataType_FLOAT16;
      } else if (s == "float") {
        return TensorProto_DataType_FLOAT;
      } else if (s == "double") {
        return TensorProto_DataType_DOUBLE;
      } else if (s == "byte") {
        return TensorProto_DataType_BYTE;
      } else if (s == "string") {
        return TensorProto_DataType_STRING;
      } else if (s == "bool") {
        return TensorProto_DataType_BOOL;
      } else if (s == "hash") {
        return TensorProto_DataType_ZERO_COLLISION_HASH;
      }
      // return default data type, float
      return TensorProto_DataType_FLOAT;
    */
}

/// Generates ShapeInfo from Blob.
#[inline] pub fn get_shape_info_from_blob(blob: *const Blob) -> ShapeInfo {
    
    todo!();
    /*
        ShapeInfo shape_info;
      shape_info.shape = GetTensorShapeOfBlob(blob);
      if (!shape_info.shape.unknown_shape()) {
        shape_info.setDimType(std::vector<TensorBoundShape::DimType>(
            shape_info.shape.dims_size(), TensorBoundShape_DimType_CONSTANT));
      }
      if (blob->meta().id() == TypeMeta::Id<int8::Int8TensorCPU>()) {
        shape_info.is_quantized = true;
        LoadInt8TensorInfoOfBlob(
            &shape_info.q_info.scale,
            &shape_info.q_info.offset,
            &shape_info.q_info.axis,
            blob);
      } else {
    #ifndef C10_MOBILE
        auto function_ptr =
            ExternalTensorFunctionsBaseRegistry()->Create(blob->meta().id());
        if (function_ptr != nullptr) {
          shape_info.is_quantized = function_ptr->isQuantized();
          function_ptr->LoadInfoOfBlob(
              blob,
              &shape_info.q_info.scale,
              &shape_info.q_info.offset,
              &shape_info.q_info.axis);
        }
    #endif
      }
      return shape_info;
    */
}

/**
  | In-place modify TensorShape's shape
  | at a specific dimension
  |
  */
#[inline] pub fn modify_tensor_shape_dim_size(
    tensor_shape: *mut TensorShape,
    dim_index:    i32,
    old_size:     i64,
    new_size:     i64)  {
    
    todo!();
    /*
        CAFFE_ENFORCE(
          old_size > 0, "Old size should be non-zero, old_size: ", old_size);
      CAFFE_ENFORCE(
          tensor_shape->dims(dim_index) % old_size == 0,
          "tensor_shape->dims[",
          dim_index,
          "] = ",
          tensor_shape->dims(dim_index),
          " cannot be divided by old_size ",
          old_size);
      int64_t modified_size = (tensor_shape->dims(dim_index) * new_size) / old_size;
      tensor_shape->set_dims(dim_index, modified_size);
    */
}

/**
  | In-place modify TensorBoundShape
  | to change shape size based on type
  |
  */
#[inline] pub fn change_tensor_bound_shapes(
    tensor_shape_and_type: &mut TensorBoundShape,
    old_batch_size:        i64,
    old_seq_size:          i64,
    new_batch_size:        i64,
    new_seq_size:          i64)  {
    
    todo!();
    /*
        CAFFE_ENFORCE(
          tensor_shape_and_type.dim_type().size() ==
          tensor_shape_and_type.shape().dims().size());

      for (int i = 0; i < tensor_shape_and_type.dim_type().size(); i++) {
        TensorBoundShape_DimType dim_type = tensor_shape_and_type.dim_type(i);
        // Need to change max_batch_size
        if (dim_type == TensorBoundShape_DimType_BATCH ||
            dim_type == TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX ||
            dim_type == TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX_DEFAULT) {
          TensorShape* tensor_shape = tensor_shape_and_type.mutable_shape();
          modifyTensorShapeDimSize(tensor_shape, i, old_batch_size, new_batch_size);
        }
        // Need to change max_seq_size
        if (dim_type == TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX_DEFAULT ||
            dim_type == TensorBoundShape_DimType_FEATURE_MAX_DEFAULT) {
          TensorShape* tensor_shape = tensor_shape_and_type.mutable_shape();
          modifyTensorShapeDimSize(tensor_shape, i, old_seq_size, new_seq_size);
        }
      }
    */
}

/**
  | Extract shape info from tensorBoundShapes to
  | a ShapeInfoMap.
  |
  | Change shape according to new max_batch_size
  | and max_feature_len at the same time if
  | necessary.
  */
#[inline] pub fn extract_shape_info_from_tensor_bound_shapes(
    tensor_bound_shapes: TensorBoundShapes,
    new_max_batch_size:  Option<i64>,
    new_max_feature_len: Option<i64>) -> ShapeInfoMap 
{

    let new_max_batch_size = new_max_batch_size.unwrap_or(-1);
    let new_max_feature_len = new_max_feature_len.unwrap_or(-1);
    
    todo!();
    /*
        ShapeInfoMap shape_info_map;
      if (new_max_batch_size == -1) {
        new_max_batch_size = tensor_bound_shapes.max_batch_size();
      }
      if (new_max_feature_len == -1) {
        new_max_feature_len = tensor_bound_shapes.max_feature_len();
      }
      for (auto& tensor_bound_shape : *(tensor_bound_shapes.mutable_shapes())) {
        std::vector<TensorBoundShape::DimType> dim_types;
        dim_types.reserve(tensor_bound_shape.shape().dims_size());
        for (auto dim_type : tensor_bound_shape.dim_type()) {
          dim_types.emplace_back(TensorBoundShape::DimType(dim_type));
        }
        changeTensorBoundShapes(
            tensor_bound_shape,
            tensor_bound_shapes.max_batch_size(),
            tensor_bound_shapes.max_feature_len(),
            new_max_batch_size,
            new_max_feature_len);
        shape_info_map[tensor_bound_shape.name()] =
            ShapeInfo(dim_types, std::move(tensor_bound_shape.shape()));
      }
      return shape_info_map;
    */
}

impl PartialEq<ShapeInfo> for ShapeInfo {
    
    fn eq(&self, other: &ShapeInfo) -> bool {
        todo!();
        /*
            return lhs.getDimType() == rhs.getDimType() &&
          lhs.shape.SerializeAsString() == rhs.shape.SerializeAsString();
        */
    }
}

impl Eq for ShapeInfo {}

/**
  | Construct a ShapeInfo instance from
  | TensorShape and constructed dimType.
  |
  | Default first dimension of dimType is BATCH,
  | reason:
  |
  | We treat first dimension of hinted shapes as
  | BATCH.
  |
  | If there are shape hints on blobs in the
  | workspace, since they are already inserted as
  | CONSTANT, it will take effect here.
  |
  | For SEQ typed tensors, there are only a few of
  | them and they will be handled by
  | BoundShapeInferencer.
  */
#[inline] pub fn construct_shape_info_with_default_dim_type(
    shape: TensorShape, 
    default_first_dim_type: Option<TensorBoundShape_DimType>) -> ShapeInfo 
{
    let default_first_dim_type = default_first_dim_type.unwrap_or(TensorBoundShape_DimType::BATCH);

    todo!();
    /*
        std::vector<TensorBoundShape_DimType> dimType(
          shape.dims_size(), TensorBoundShape_DimType_CONSTANT);
      if (dimType.size()) {
        dimType[0] = defaultFirstDimType;
      }
      return ShapeInfo(dimType, shape);
    */
}

#[inline] pub fn parse_shape_info_map_from_string(
    input:       &String, 
    shape_hints: &mut ShapeInfoMap)
{
    todo!();
    /*
        auto hints = caffe2::split('#', input);
      for (const auto& hint : hints) {
        auto kv = caffe2::split(',', hint);
        CAFFE_ENFORCE_GE(kv.size(), 2, "Cannot parse shape hint: ", hint);
        const auto& name = kv[0];

        TensorShape shape;
        size_t size = kv.size();
        CAFFE_ENFORCE_GT(size, 1);
        if (!isNumber(kv[size - 1])) {
          // last value is the type
          shape.set_data_type(toTensorProtoDataType(kv[size - 1]));
          size--;
        } else {
          if (name.find("int8") != std::string::npos) {
            // Kept for backwards compatibility.
            // Set type explicitly to overwrite it.
            shape.set_data_type(TensorProto_DataType_UINT8);
          } else {
            shape.set_data_type(TensorProto_DataType_FLOAT);
          }
        }

        bool valid = true;
        for (int i = 1; i < size; i++) {
          auto dim = kv[i];
          try {
            shape.add_dims(std::stoi(dim));
          } catch (const std::exception& e) {
            valid = false;
            CAFFE_THROW("Cannot parse shape hint: ", hint);
          }
        }
        if (valid) {
          shape_hints.emplace(name, constructShapeInfoWithDefaultDimType(shape));
        }
      }
    */
}
