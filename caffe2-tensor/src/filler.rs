crate::ix!();

/**
  | TODO: replace filler distribution
  | enum with a better abstraction
  |
  */
pub enum FillerDistribution { 
    FD_UNIFORM, 
    FD_FIXEDSUM, 
    FD_SYNTHETIC 
}

pub struct TensorFiller {

    shape:        Vec<i64>,

    /**
      | TODO: type is unknown until a user starts
      | to fill data; cast everything to double
      | for now.
      |
      */
    min:          f64, // default = 0.0
    max:          f64, // default = 1.0
    dist:         FillerDistribution,
    fixed_sum:    f64,
}

impl Default for TensorFiller {
    
    fn default() -> Self {
        todo!();
        /*
            : TensorFiller(std::vector<int64_t>()
        */
    }
}

impl TensorFiller {

    #[inline] pub fn fill<'a,Type, Context>(
        &mut self, 
        tensor:  *mut Tensor,
        context: *mut Context) 
    {
        todo!();
        /*
            CAFFE_ENFORCE(context, "context is null");
            CAFFE_ENFORCE(tensor, "tensor is null");
            auto min = (min_ < Type::min)
                ? Type::min
                : static_cast<Type>(min_);
            auto max = (max_ > Type::max)
                ? Type::max
                : static_cast<Type>(max_);
            CAFFE_ENFORCE_LE(min, max);

            Tensor temp_tensor(shape_, Context::GetDeviceType());
            std::swap(*tensor, temp_tensor);
            Type* data = tensor->template mutable_data<Type>();

            // select distribution
            switch (dist_) {
              case FD_UNIFORM: {
                math::RandUniform<Type, Context>(
                    tensor->numel(), min, max, data, context);
                break;
              }
              case FD_FIXEDSUM: {
                auto fixed_sum = static_cast<Type>(fixed_sum_);
                CAFFE_ENFORCE_LE(min * tensor->numel(), fixed_sum);
                CAFFE_ENFORCE_GE(max * tensor->numel(), fixed_sum);
                math::RandFixedSum<Type, Context>(
                    tensor->numel(), min, max, fixed_sum_, data, context);
                break;
              }
              case FD_SYNTHETIC: {
                math::RandSyntheticData<Type, Context>(
                    tensor->numel(), min, max, data, context);
                break;
              }
            }
        */
    }
    
    #[inline] pub fn dist<'a>(&mut self, dist: FillerDistribution) -> &'a mut TensorFiller {
        
        todo!();
        /*
            dist_ = dist;
        return *this;
        */
    }

    #[inline] pub fn min<'a,Type>(&mut self, min: Type) -> &'a mut TensorFiller {
        todo!();
        /*
            min_ = (double)min;
            return *this;
        */
    }

    #[inline] pub fn max<'a,Type>(&mut self, max: Type) -> &'a mut TensorFiller {
        todo!();
        /*
            max_ = (double)max;
            return *this;
        */
    }

    #[inline] pub fn fixed_sum<'a,Type>(&mut self, fixed_sum: Type) -> &'a mut TensorFiller {
        todo!();
        /*
            dist_ = FD_FIXEDSUM;
            fixed_sum_ = (double)fixed_sum;
            return *this;
        */
    }

    /**
      | A helper function to construct the lengths
      | vector for sparse features
      |
      | We try to pad least one index per batch
      | unless the total_length is 0
      */
    #[inline] pub fn sparse_lengths<'a,Type>(&mut self, total_length: Type) -> &'a mut TensorFiller {
        todo!();
        /*
            return FixedSum(total_length)
                .Min(std::min(static_cast<Type>(1), total_length))
                .Max(total_length);
        */
    }

    /**
      | a helper function to construct the segments
      | vector for sparse features
      |
      */
    #[inline] pub fn sparse_segments<'a,Type>(&mut self, max_segment: Type) -> &'a mut TensorFiller {
        todo!();
        /*
            CAFFE_ENFORCE(dist_ != FD_FIXEDSUM);
        return Min(0).Max(max_segment).Dist(FD_SYNTHETIC);
        */
    }
    
    #[inline] pub fn shape<'a>(&mut self, shape: &Vec<i64>) -> &'a mut TensorFiller {
        
        todo!();
        /*
            shape_ = shape;
        return *this;
        */
    }
    
    pub fn new<Type>(shape: &Vec<i64>, fixed_sum: Type) -> Self {
        todo!();
        /*
            : shape_(shape), dist_(FD_FIXEDSUM), fixed_sum_((double)fixed_sum)
        */
    }
    
    pub fn new_from_shape(shape: &Vec<i64>) -> Self {
        todo!();
        /*
            : shape_(shape), dist_(FD_UNIFORM), fixed_sum_(0)
        */
    }
    
    #[inline] pub fn debug_string(&self) -> String {
        
        todo!();
        /*
            std::stringstream stream;
        stream << "shape = [" << shape_ << "]; min = " << min_
               << "; max = " << max_;
        switch (dist_) {
          case FD_FIXEDSUM:
            stream << "; dist = FD_FIXEDSUM";
            break;
          case FD_SYNTHETIC:
            stream << "; dist = FD_SYNTHETIC";
            break;
          default:
            stream << "; dist = FD_UNIFORM";
            break;
        }
        return stream.str();
        */
    }
}
