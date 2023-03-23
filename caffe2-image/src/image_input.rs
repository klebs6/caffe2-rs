crate::ix!();

/**
  | Imports and processes images from a
  | database. For each run of the operator,
  | batch_size images will be processed.
  | GPUs can optionally be used for part
  | of the processing.
  | 
  | The following transformations are
  | applied to the image
  | 
  | - A bounding box is applied to the initial
  | image (optional)
  | 
  | - The image is rescaled either up or down
  | (with the scale argument) or just up
  | (with the minsize argument)
  | 
  | - The image is randomly cropped (crop
  | size is passed as an argument but the
  | location of the crop is random except
  | if is_test is passed in which case the
  | image in cropped at the center)
  | 
  | - The image is normalized. Each of its
  | color channels can have separate normalization
  | values
  | 
  | The dimension of the output image will
  | always be cropxcrop
  |
  */
pub struct ImageInputOp<Context> {
    base: PrefetchOperator<Context>,

    owned_reader:                            Box<DBReader>,
    reader:                                  *const DBReader,

    prefetched_image:                        Tensor,
    prefetched_label:                        Tensor,
    prefetched_additional_outputs:           Vec<Tensor>,
    prefetched_image_on_device:              Tensor,
    prefetched_label_on_device:              Tensor,
    prefetched_additional_outputs_on_device: Vec<Tensor>,

    /// Default parameters for images
    default_arg:                             PerImageArg,
    batch_size:                              i32,
    label_type:                              LabelType,
    num_labels:                              i32,

    color:                                   bool,
    color_jitter:                            bool,
    img_saturation:                          f32,
    img_brightness:                          f32,
    img_contrast:                            f32,
    color_lighting:                          bool,
    color_lighting_std:                      f32,
    color_lighting_eigvecs:                  Vec<Vec<f32>>,
    color_lighting_eigvals:                  Vec<f32>,
    scale_jitter_type:                       ScaleJitterType,
    scale:                                   i32,

    /**
      | Minsize is similar to scale except that it
      | will only force the image to scale up if
      | it is too small. In other words, it
      | ensures that both dimensions of the image
      | are at least minsize_
      */
    minsize:                                 i32,
    warp:                                    bool,
    crop:                                    i32,
    mean:                                    Vec<f32>,
    std:                                     Vec<f32>,
    mean_gpu:                                Tensor,
    std_gpu:                                 Tensor,
    mirror:                                  bool,

    /**
      | Set to 1 to do deterministic cropping.
      | Defaults to 0
      |
      */
    is_test:                                 bool,

    use_caffe_datum:                         bool,
    gpu_transform:                           bool,

    mean_std_copied:                         bool, // = false;

    /// thread pool for parse + decode
    num_decode_threads:                      i32,
    additional_inputs_offset:                i32,
    additional_inputs_count:                 i32,
    additional_output_sizes:                 Vec<i32>,
    thread_pool:                             Arc<TaskThreadPool>,

    /// Output type for GPU transform path
    output_type:                             TensorProto_DataType,

    /// random minsize
    random_scale:                            Vec<i32>,
    random_scaling:                          bool,

    /// Working variables
    randgen_per_thread:                      Vec<mt19937::MT19937>,

    /**
      | number of exceptions produced by opencv
      | while reading image data
      |
      */
    num_decode_errors_in_batch:              Atomic<i64>, //{0};

    /// opencv exceptions tolerance
    max_decode_error_ratio:                  f32,
}

num_inputs!{ImageInput, (0,1)}

num_outputs!{ImageInput, (2,INT_MAX)}

inputs!{ImageInput, 
    0 => ("reader",                  "The input reader (a db::DBReader)")
}

outputs!{ImageInput, 
    0 => ("data",                    "Tensor containing the images"),
    1 => ("label",                   "Tensor containing the labels"),
    2 => ("additional outputs",      "Any outputs after the first 2 will be Tensors read from the input TensorProtos")
}

args!{ImageInput, 
    0  => ("batch_size",             "Number of images to output for each run of the operator. Must be 1 or greater"),
    1  => ("color",                  "Number of color channels (1 or 3). Defaults to 1"),
    2  => ("color_jitter",           "Whether or not to do color jitter. Defaults to 0"),
    3  => ("img_saturation",         "Image saturation scale used in color jittering. Defaults to 0.4"),
    4  => ("img_brightness",         "Image brightness scale used in color jittering. Defaults to 0.4"),
    5  => ("img_contrast",           "Image contrast scale used in color jittering. Defaults to 0.4"),
    6  => ("color_lighting",         "Whether or not to do color lighting. Defaults to 0"),
    7  => ("color_lighting_std",     "Std of normal distribution where color lighting scaling factor is sampled. Defaults to 0.1"),
    8  => ("scale_jitter_type",      "Type 0: No scale jittering Type 1: Inception-style scale jittering"),
    9  => ("label_type",             "Type 0: single integer label for multi-class classification. Type 1: sparse active label indices for multi-label classification. Type 2: dense label embedding vector for label embedding regression"),
    10 => ("scale",                  "Scale the size of the smallest dimension of the image to this. Scale and minsize are mutually exclusive. Must be larger than crop"),
    11 => ("minsize",                "Scale the size of the smallest dimension of the image to this only if the size is initially smaller. Scale and minsize are mutually exclusive. Must be larger than crop."),
    12 => ("warp",                   "If 1, both dimensions of the image will be set to minsize or scale; otherwise, the other dimension is proportionally scaled. Defaults to 0"),
    13 => ("crop",                   "Size to crop the image to. Must be provided"),
    14 => ("mirror",                 "Whether or not to mirror the image. Defaults to 0"),
    15 => ("mean",                   "Mean by which to normalize color channels. Defaults to 0."),
    16 => ("mean_per_channel",       "Vector of means per color channel  (1 or 3 elements). Defaults to mean argument. Channel order BGR"),
    17 => ("std",                    "Standard deviation by which to normalize color channels. Defaults to 1."),
    18 => ("std_per_channel",        "Vector of standard dev. per color channel  (1 or 3 elements). Defaults to std argument. Channel order is BGR"),
    19 => ("bounding_ymin",          "Bounding box coordinate. Defaults to -1 (none)"),
    20 => ("bounding_xmin",          "Bounding box coordinate. Defaults to -1 (none)"),
    21 => ("bounding_height",        "Bounding box coordinate. Defaults to -1 (none)"),
    22 => ("bounding_width",         "Bounding box coordinate. Defaults to -1 (none)"),
    23 => ("use_caffe_datum",        "1 if the input is in Caffe format. Defaults to 0"),
    24 => ("use_gpu_transform",      "1 if GPU acceleration should be used. Defaults to 0. Can only be 1 in a CUDAContext"),
    25 => ("decode_threads",         "Number of CPU decode/transform threads. Defaults to 4"),
    26 => ("output_type",            "If gpu_transform, can set to FLOAT or FLOAT16."),
    27 => ("db",                     "Name of the database (if not passed as input)"),
    28 => ("db_type",                "Type of database (if not passed as input). Defaults to leveldb"),
    29 => ("output_sizes",           "The sizes of any outputs besides the data and label (should have a number of elements equal to the number of additional outputs)"),
    30 => ("random_scale",           "[min, max] shortest-side desired for image resize. Defaults to [-1, -1] or no random resize desired.")
}

tensor_inference_function!{ImageInput, /* [](const OperatorDef& def,
                                const vector<TensorShape>& /* unused */) {
      vector<TensorShape> out(2);
      ArgumentHelper helper(def);
      int batch_size = helper.GetSingleArgument<int>("batch_size", 0);
      int crop = helper.GetSingleArgument<int>("crop", -1);
      int color = helper.GetSingleArgument<int>("color", 1);
      CHECK_GT(crop, 0);
      out[0] = CreateTensorShape(
          vector<int>{batch_size, crop, crop, color ? 3 : 1},
          TensorProto::FLOAT);
      out[1] =
          CreateTensorShape(vector<int>{1, batch_size}, TensorProto::INT32);
      return out;
    } */}


no_gradient!{ImageInput}

impl<Context> Drop for ImageInputOp<Context> {
    fn drop(&mut self) {
        todo!();
        //PrefetchOperator<Context>::Finalize();
    }
}

impl<Context> ImageInputOp<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : PrefetchOperator<Context>(operator_def, ws),
          reader_(nullptr),
          batch_size_( OperatorStorage::template GetSingleArgument<int>("batch_size", 0)),
          label_type_(static_cast<LABEL_TYPE>( OperatorStorage::template GetSingleArgument<int>("label_type", 0))),
          num_labels_( OperatorStorage::template GetSingleArgument<int>("num_labels", 0)),
          color_(OperatorStorage::template GetSingleArgument<int>("color", 1)),
          color_jitter_( OperatorStorage::template GetSingleArgument<int>("color_jitter", 0)),
          img_saturation_(OperatorStorage::template GetSingleArgument<float>( "img_saturation", 0.4)),
          img_brightness_(OperatorStorage::template GetSingleArgument<float>( "img_brightness", 0.4)),
          img_contrast_( OperatorStorage::template GetSingleArgument<float>("img_contrast", 0.4)),
          color_lighting_( OperatorStorage::template GetSingleArgument<int>("color_lighting", 0)),
          color_lighting_std_(OperatorStorage::template GetSingleArgument<float>( "color_lighting_std", 0.1)),
          scale_jitter_type_(static_cast<SCALE_JITTER_TYPE>( OperatorStorage::template GetSingleArgument<int>( "scale_jitter_type", 0))),
          scale_(OperatorStorage::template GetSingleArgument<int>("scale", -1)),
          minsize_(OperatorStorage::template GetSingleArgument<int>("minsize", -1)),
          warp_(OperatorStorage::template GetSingleArgument<int>("warp", 0)),
          crop_(OperatorStorage::template GetSingleArgument<int>("crop", -1)),
          mirror_(OperatorStorage::template GetSingleArgument<int>("mirror", 0)),
          is_test_(OperatorStorage::template GetSingleArgument<int>( OpSchema::Arg_IsTest, 0)),
          use_caffe_datum_( OperatorStorage::template GetSingleArgument<int>("use_caffe_datum", 0)),
          gpu_transform_(OperatorStorage::template GetSingleArgument<int>( "use_gpu_transform", 0)),
          num_decode_threads_( OperatorStorage::template GetSingleArgument<int>("decode_threads", 4)),
          additional_output_sizes_( OperatorStorage::template GetRepeatedArgument<int>("output_sizes", {})),
          thread_pool_(std::make_shared<TaskThreadPool>(num_decode_threads_)),
          // output type only supported with CUDA and use_gpu_transform for now
          output_type_( cast::GetCastDataType(ArgumentHelper(operator_def), "output_type")),
          random_scale_(OperatorStorage::template GetRepeatedArgument<int>( "random_scale", {-1, -1})),
          max_decode_error_ratio_(OperatorStorage::template GetSingleArgument<float>( "max_decode_error_ratio", 1.0)) 

        if ((random_scale_[0] == -1) || (random_scale_[1] == -1)) {
            random_scaling_ = false;
        } else {
            random_scaling_ = true;
            minsize_ = random_scale_[0];
        }

        mean_ = OperatorStorage::template GetRepeatedArgument<float>(
            "mean_per_channel",
            {OperatorStorage::template GetSingleArgument<float>("mean", 0.)});

        std_ = OperatorStorage::template GetRepeatedArgument<float>(
            "std_per_channel",
            {OperatorStorage::template GetSingleArgument<float>("std", 1.)});

        if (additional_output_sizes_.size() == 0) {
            additional_output_sizes_ = std::vector<int>(OutputSize() - 2, 1);
        } else {
            CAFFE_ENFORCE(
                additional_output_sizes_.size() == OutputSize() - 2,
                "If the output sizes are specified, they must be specified for all "
                "additional outputs");
        }
        additional_inputs_count_ = OutputSize() - 2;

        default_arg_.bounding_params = {
            false,
            OperatorStorage::template GetSingleArgument<int>("bounding_ymin", -1),
            OperatorStorage::template GetSingleArgument<int>("bounding_xmin", -1),
            OperatorStorage::template GetSingleArgument<int>("bounding_height", -1),
            OperatorStorage::template GetSingleArgument<int>("bounding_width", -1),
        };

        if (operator_def.input_size() == 0) {
            LOG(ERROR) << "You are using an old ImageInputOp format that creates "
                "a local db reader. Consider moving to the new style "
                "that takes in a DBReader blob instead.";
            string db_name = OperatorStorage::template GetSingleArgument<string>("db", "");
            CAFFE_ENFORCE_GT(db_name.size(), 0, "Must specify a db name.");
            owned_reader_.reset(new db::DBReader(
                    OperatorStorage::template GetSingleArgument<string>("db_type", "leveldb"),
                    db_name));
            reader_ = owned_reader_.get();
        }

        // hard-coded PCA eigenvectors and eigenvalues, based on RBG channel order
        color_lighting_eigvecs_.push_back(
            std::vector<float>{-144.7125f, 183.396f, 102.2295f});
        color_lighting_eigvecs_.push_back(
            std::vector<float>{-148.104f, -1.1475f, -207.57f});
        color_lighting_eigvecs_.push_back(
            std::vector<float>{-148.818f, -177.174f, 107.1765f});

        color_lighting_eigvals_ = std::vector<float>{0.2175f, 0.0188f, 0.0045f};

        CAFFE_ENFORCE_GT(batch_size_, 0, "Batch size should be nonnegative.");
        if (use_caffe_datum_) {
            CAFFE_ENFORCE(
                label_type_ == SINGLE_LABEL || label_type_ == SINGLE_LABEL_WEIGHTED,
                "Caffe datum only supports single integer label");
        }
        if (label_type_ != SINGLE_LABEL && label_type_ != SINGLE_LABEL_WEIGHTED) {
            CAFFE_ENFORCE_GT(
                num_labels_,
                0,
                "Number of labels must be set for using either sparse label indices or dense label embedding.");
        }
        if (label_type_ == MULTI_LABEL_WEIGHTED_SPARSE ||
            label_type_ == SINGLE_LABEL_WEIGHTED) {
            additional_inputs_offset_ = 3;
        } else {
            additional_inputs_offset_ = 2;
        }
        CAFFE_ENFORCE(
            (scale_ > 0) != (minsize_ > 0),
            "Must provide one and only one of scaling or minsize");
        CAFFE_ENFORCE_GT(crop_, 0, "Must provide the cropping value.");
        CAFFE_ENFORCE_GE(
            scale_ > 0 ? scale_ : minsize_,
            crop_,
            "The scale/minsize value must be no smaller than the crop value.");

        CAFFE_ENFORCE_EQ(
            mean_.size(),
            std_.size(),
            "The mean and std. dev vectors must be of the same size.");
        CAFFE_ENFORCE(
            mean_.size() == 1 || mean_.size() == 3,
            "The mean and std. dev vectors must be of size 1 or 3");
        CAFFE_ENFORCE(
            !use_caffe_datum_ || OutputSize() == 2,
            "There can only be 2 outputs if the Caffe datum format is used");

        CAFFE_ENFORCE(
            random_scale_.size() == 2, "Must provide [scale_min, scale_max]");
        CAFFE_ENFORCE_GE(
            random_scale_[1],
            random_scale_[0],
            "random scale must provide a range [min, max]");

        if (default_arg_.bounding_params.ymin < 0 ||
            default_arg_.bounding_params.xmin < 0 ||
            default_arg_.bounding_params.height < 0 ||
            default_arg_.bounding_params.width < 0) {
            default_arg_.bounding_params.valid = false;
        } else {
            default_arg_.bounding_params.valid = true;
        }

        if (mean_.size() == 1) {
            // We are going to extend to 3 using the first value
            mean_.resize(3, mean_[0]);
            std_.resize(3, std_[0]);
        }

        LOG(INFO) << "Creating an image input op with the following setting: ";
        LOG(INFO) << "    Using " << num_decode_threads_ << " CPU threads;";
        if (gpu_transform_) {
            LOG(INFO) << "    Performing transformation on GPU";
        }
        LOG(INFO) << "    Outputting in batches of " << batch_size_ << " images;";
        LOG(INFO) << "    Treating input image as "
            << (color_ ? "color " : "grayscale ") << "image;";
        if (default_arg_.bounding_params.valid) {
            LOG(INFO) << "    Applying a default bounding box of Y ["
                << default_arg_.bounding_params.ymin << "; "
                << default_arg_.bounding_params.ymin +
                default_arg_.bounding_params.height
                << ") x X [" << default_arg_.bounding_params.xmin << "; "
                << default_arg_.bounding_params.xmin +
                default_arg_.bounding_params.width
                << ")";
        }
        if (scale_ > 0 && !random_scaling_) {
            LOG(INFO) << "    Scaling image to " << scale_
                << (warp_ ? " with " : " without ") << "warping;";
        } else {
            if (random_scaling_) {
                // randomly set min_size_ for each image
                LOG(INFO) << "    Randomly scaling shortest side between "
                    << random_scale_[0] << " and " << random_scale_[1];
            } else {
                // Here, minsize_ > 0
                LOG(INFO) << "    Ensuring minimum image size of " << minsize_
                    << (warp_ ? " with " : " without ") << "warping;";
            }
        }
        LOG(INFO) << "    " << (is_test_ ? "Central" : "Random")
            << " cropping image to " << crop_
            << (mirror_ ? " with " : " without ") << "random mirroring;";
        LOG(INFO) << "Label Type: " << label_type_;
        LOG(INFO) << "Num Labels: " << num_labels_;

        auto mit = mean_.begin();
        auto sit = std_.begin();

        for (int i = 0; mit != mean_.end() && sit != std_.end(); ++mit, ++sit, ++i) {
            LOG(INFO) << "    Default [Channel " << i << "] Subtract mean " << *mit
                << " and divide by std " << *sit << ".";
            // We actually will use the inverse of std, so inverse it here
            *sit = 1.f / *sit;
        }
        LOG(INFO) << "    Outputting images as "
            << OperatorStorage::template GetSingleArgument<string>(
                "output_type", "unknown")
            << ".";

        std::mt19937 meta_randgen(time(nullptr));
        for (int i = 0; i < num_decode_threads_; ++i) {
            randgen_per_thread_.emplace_back(meta_randgen());
        }
        ReinitializeTensor(
            &prefetched_image_,
            {int64_t(batch_size_),
            int64_t(crop_),
            int64_t(crop_),
            int64_t(color_ ? 3 : 1)},
            at::dtype<uint8_t>().device(CPU));
        std::vector<int64_t> sizes;
        if (label_type_ != SINGLE_LABEL && label_type_ != SINGLE_LABEL_WEIGHTED) {
            sizes = std::vector<int64_t>{int64_t(batch_size_), int64_t(num_labels_)};
        } else {
            sizes = std::vector<int64_t>{batch_size_};
        }
        // data type for prefetched_label_ is actually not known here..
        ReinitializeTensor(&prefetched_label_, sizes, at::dtype<int>().device(CPU));

        for (int i = 0; i < additional_output_sizes_.size(); ++i) {
            prefetched_additional_outputs_on_device_.emplace_back();
            prefetched_additional_outputs_.emplace_back();
        }
        */
    }

    #[inline] pub fn get_image_and_label_and_info_from_dbvalue(
        &mut self, 
        value:    &String,
        img:      *mut cv::Mat,
        info:     &mut PerImageArg,
        item_id:  i32,
        randgen:  *mut mt19937::MT19937) -> bool 
    {
        todo!();
        /*
            //
      // recommend using --caffe2_use_fatal_for_enforce=1 when using ImageInputOp
      // as this function runs on a worker thread and the exceptions from
      // CAFFE_ENFORCE are silently dropped by the thread worker functions
      //
      cv::Mat src;

      // Use the default information for images
      info = default_arg_;
      if (use_caffe_datum_) {
        // The input is a caffe datum format.
        CaffeDatum datum;
        CAFFE_ENFORCE(datum.ParseFromString(value));

        prefetched_label_.mutable_data<int>()[item_id] = datum.label();
        if (datum.encoded()) {
          // encoded image in datum.
          // count the number of exceptions from opencv imdecode
          try {
            src = cv::imdecode(
                cv::Mat(
                    1,
                    datum.data().size(),
                    CV_8UC1,
                    const_cast<char*>(datum.data().data())),
                color_ ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE);
            if (src.rows == 0 || src.cols == 0) {
              num_decode_errors_in_batch_++;
              src = cv::Mat::zeros(cv::Size(224, 224), CV_8UC3);
            }
          } catch (cv::Exception& e) {
            num_decode_errors_in_batch_++;
            src = cv::Mat::zeros(cv::Size(224, 224), CV_8UC3);
          }
        } else {
          // Raw image in datum.
          CAFFE_ENFORCE(datum.channels() == 3 || datum.channels() == 1);

          int src_c = datum.channels();
          src.create(
              datum.height(), datum.width(), (src_c == 3) ? CV_8UC3 : CV_8UC1);

          if (src_c == 1) {
            memcpy(src.ptr<uchar>(0), datum.data().data(), datum.data().size());
          } else {
            // Datum stores things in CHW order, let's do HWC for images to make
            // things more consistent with conventional image storage.
            for (int c = 0; c < 3; ++c) {
              const char* datum_buffer =
                  datum.data().data() + datum.height() * datum.width() * c;
              uchar* ptr = src.ptr<uchar>(0) + c;
              for (int h = 0; h < datum.height(); ++h) {
                for (int w = 0; w < datum.width(); ++w) {
                  *ptr = *(datum_buffer++);
                  ptr += 3;
                }
              }
            }
          }
        }
      } else {
        // The input is a caffe2 format.
        TensorProtos protos;
        CAFFE_ENFORCE(protos.ParseFromString(value));
        const TensorProto& image_proto = protos.protos(0);
        const TensorProto& label_proto = protos.protos(1);
        // add handle protos
        vector<TensorProto> additional_output_protos;
        int start = additional_inputs_offset_;
        int end = start + additional_inputs_count_;
        for (int i = start; i < end; ++i) {
          additional_output_protos.push_back(protos.protos(i));
        }

        if (protos.protos_size() == end + 1) {
          // We have bounding box information
          const TensorProto& bounding_proto = protos.protos(end);
          DCHECK_EQ(bounding_proto.data_type(), TensorProto::INT32);
          DCHECK_EQ(bounding_proto.int32_data_size(), 4);
          info.bounding_params.valid = true;
          info.bounding_params.ymin = bounding_proto.int32_data(0);
          info.bounding_params.xmin = bounding_proto.int32_data(1);
          info.bounding_params.height = bounding_proto.int32_data(2);
          info.bounding_params.width = bounding_proto.int32_data(3);
        }

        if (image_proto.data_type() == TensorProto::STRING) {
          // encoded image string.
          DCHECK_EQ(image_proto.string_data_size(), 1);
          const string& encoded_image_str = image_proto.string_data(0);
          int encoded_size = encoded_image_str.size();
          // We use a cv::Mat to wrap the encoded str so we do not need a copy.
          // count the number of exceptions from opencv imdecode
          try {
            src = cv::imdecode(
                cv::Mat(
                    1,
                    &encoded_size,
                    CV_8UC1,
                    const_cast<char*>(encoded_image_str.data())),
                color_ ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE);
            if (src.rows == 0 || src.cols == 0) {
              num_decode_errors_in_batch_++;
              src = cv::Mat::zeros(cv::Size(224, 224), CV_8UC3);
            }
          } catch (cv::Exception& e) {
            num_decode_errors_in_batch_++;
            src = cv::Mat::zeros(cv::Size(224, 224), CV_8UC3);
          }
        } else if (image_proto.data_type() == TensorProto::BYTE) {
          // raw image content.
          int src_c = (image_proto.dims_size() == 3) ? image_proto.dims(2) : 1;
          CAFFE_ENFORCE(src_c == 3 || src_c == 1);

          src.create(
              image_proto.dims(0),
              image_proto.dims(1),
              (src_c == 3) ? CV_8UC3 : CV_8UC1);
          memcpy(
              src.ptr<uchar>(0),
              image_proto.byte_data().data(),
              image_proto.byte_data().size());
        } else {
          LOG(FATAL) << "Unknown image data type.";
        }

        // TODO: if image decoding was unsuccessful, set label to 0
        if (label_proto.data_type() == TensorProto::FLOAT) {
          if (label_type_ == SINGLE_LABEL || label_type_ == SINGLE_LABEL_WEIGHTED) {
            DCHECK_EQ(label_proto.float_data_size(), 1);
            prefetched_label_.mutable_data<float>()[item_id] =
                label_proto.float_data(0);
          } else if (label_type_ == MULTI_LABEL_SPARSE) {
            float* label_data =
                prefetched_label_.mutable_data<float>() + item_id * num_labels_;
            memset(label_data, 0, sizeof(float) * num_labels_);
            for (int i = 0; i < label_proto.float_data_size(); ++i) {
              label_data[(int)label_proto.float_data(i)] = 1.0;
            }
          } else if (label_type_ == MULTI_LABEL_WEIGHTED_SPARSE) {
            const TensorProto& weight_proto = protos.protos(2);
            float* label_data =
                prefetched_label_.mutable_data<float>() + item_id * num_labels_;
            memset(label_data, 0, sizeof(float) * num_labels_);
            for (int i = 0; i < label_proto.float_data_size(); ++i) {
              label_data[(int)label_proto.float_data(i)] =
                  weight_proto.float_data(i);
            }
          } else if (
              label_type_ == MULTI_LABEL_DENSE || label_type_ == EMBEDDING_LABEL) {
            CAFFE_ENFORCE(label_proto.float_data_size() == num_labels_);
            float* label_data =
                prefetched_label_.mutable_data<float>() + item_id * num_labels_;
            for (int i = 0; i < label_proto.float_data_size(); ++i) {
              label_data[i] = label_proto.float_data(i);
            }
          } else {
            LOG(ERROR) << "Unknown label type:" << label_type_;
          }
        } else if (label_proto.data_type() == TensorProto::INT32) {
          if (label_type_ == SINGLE_LABEL || label_type_ == SINGLE_LABEL_WEIGHTED) {
            DCHECK_EQ(label_proto.int32_data_size(), 1);
            prefetched_label_.mutable_data<int>()[item_id] =
                label_proto.int32_data(0);
          } else if (label_type_ == MULTI_LABEL_SPARSE) {
            int* label_data =
                prefetched_label_.mutable_data<int>() + item_id * num_labels_;
            memset(label_data, 0, sizeof(int) * num_labels_);
            for (int i = 0; i < label_proto.int32_data_size(); ++i) {
              label_data[label_proto.int32_data(i)] = 1;
            }
          } else if (label_type_ == MULTI_LABEL_WEIGHTED_SPARSE) {
            const TensorProto& weight_proto = protos.protos(2);
            float* label_data =
                prefetched_label_.mutable_data<float>() + item_id * num_labels_;
            memset(label_data, 0, sizeof(float) * num_labels_);
            for (int i = 0; i < label_proto.int32_data_size(); ++i) {
              label_data[label_proto.int32_data(i)] = weight_proto.float_data(i);
            }
          } else if (
              label_type_ == MULTI_LABEL_DENSE || label_type_ == EMBEDDING_LABEL) {
            CAFFE_ENFORCE(label_proto.int32_data_size() == num_labels_);
            int* label_data =
                prefetched_label_.mutable_data<int>() + item_id * num_labels_;
            for (int i = 0; i < label_proto.int32_data_size(); ++i) {
              label_data[i] = label_proto.int32_data(i);
            }
          } else {
            LOG(ERROR) << "Unknown label type:" << label_type_;
          }
        } else {
          LOG(FATAL) << "Unsupported label data type.";
        }

        for (int i = 0; i < additional_output_protos.size(); ++i) {
          auto additional_output_proto = additional_output_protos[i];
          if (additional_output_proto.data_type() == TensorProto::FLOAT) {
            float* additional_output =
                prefetched_additional_outputs_[i].template mutable_data<float>() +
                item_id * additional_output_proto.float_data_size();

            for (int j = 0; j < additional_output_proto.float_data_size(); ++j) {
              additional_output[j] = additional_output_proto.float_data(j);
            }
          } else if (additional_output_proto.data_type() == TensorProto::INT32) {
            int* additional_output =
                prefetched_additional_outputs_[i].template mutable_data<int>() +
                item_id * additional_output_proto.int32_data_size();

            for (int j = 0; j < additional_output_proto.int32_data_size(); ++j) {
              additional_output[j] = additional_output_proto.int32_data(j);
            }
          } else if (additional_output_proto.data_type() == TensorProto::INT64) {
            int64_t* additional_output =
                prefetched_additional_outputs_[i].template mutable_data<int64_t>() +
                item_id * additional_output_proto.int64_data_size();

            for (int j = 0; j < additional_output_proto.int64_data_size(); ++j) {
              additional_output[j] = additional_output_proto.int64_data(j);
            }
          } else if (additional_output_proto.data_type() == TensorProto::UINT8) {
            uint8_t* additional_output =
                prefetched_additional_outputs_[i].template mutable_data<uint8_t>() +
                item_id * additional_output_proto.int32_data_size();

            for (int j = 0; j < additional_output_proto.int32_data_size(); ++j) {
              additional_output[j] =
                  static_cast<uint8_t>(additional_output_proto.int32_data(j));
            }
          } else {
            LOG(FATAL) << "Unsupported output type.";
          }
        }
      }

      //
      // convert source to the color format requested from Op
      //
      int out_c = color_ ? 3 : 1;
      if (out_c == src.channels()) {
        *img = src;
      } else {
        cv::cvtColor(
            src, *img, (out_c == 1) ? cv::COLOR_BGR2GRAY : cv::COLOR_GRAY2BGR);
      }

      // Note(Yangqing): I believe that the mat should be created continuous.
      CAFFE_ENFORCE(img->isContinuous());

      // Sanity check now that we decoded everything

      // Ensure that the bounding box is legit
      if (info.bounding_params.valid &&
          (src.rows < info.bounding_params.ymin + info.bounding_params.height ||
           src.cols < info.bounding_params.xmin + info.bounding_params.width)) {
        info.bounding_params.valid = false;
      }

      // Apply the bounding box if requested
      if (info.bounding_params.valid) {
        // If we reach here, we know the parameters are sane
        cv::Rect bounding_box(
            info.bounding_params.xmin,
            info.bounding_params.ymin,
            info.bounding_params.width,
            info.bounding_params.height);
        *img = (*img)(bounding_box);

        /*
        LOG(INFO) << "Did bounding with ymin:"
                  << info.bounding_params.ymin << " xmin:" <<
        info.bounding_params.xmin
                  << " height:" << info.bounding_params.height
                  << " width:" << info.bounding_params.width << "\n";
        LOG(INFO) << "Bounded matrix: " << img;
        */
      } else {
        // LOG(INFO) << "No bounding\n";
      }

      cv::Mat scaled_img;
      bool inception_scale_jitter = false;
      if (scale_jitter_type_ == INCEPTION_STYLE) {
        if (!is_test_) {
          // Inception-stype scale jittering is only used for training
          inception_scale_jitter =
              RandomSizedCropping<Context>(img, crop_, randgen);
          // if a random crop is still not found, do simple random cropping later
        }
      }

      if ((scale_jitter_type_ == NO_SCALE_JITTER) ||
          (scale_jitter_type_ == INCEPTION_STYLE && !inception_scale_jitter)) {
        int scaled_width, scaled_height;
        int scale_to_use = scale_ > 0 ? scale_ : minsize_;

        // set the random minsize
        if (random_scaling_) {
          scale_to_use = std::uniform_int_distribution<>(
              random_scale_[0], random_scale_[1])(*randgen);
        }

        if (warp_) {
          scaled_width = scale_to_use;
          scaled_height = scale_to_use;
        } else if (img->rows > img->cols) {
          scaled_width = scale_to_use;
          scaled_height = static_cast<float>(img->rows) * scale_to_use / img->cols;
        } else {
          scaled_height = scale_to_use;
          scaled_width = static_cast<float>(img->cols) * scale_to_use / img->rows;
        }
        if ((scale_ > 0 &&
             (scaled_height != img->rows || scaled_width != img->cols)) ||
            (scaled_height > img->rows || scaled_width > img->cols)) {
          // We rescale in all cases if we are using scale_
          // but only to make the image bigger if using minsize_
          /*
          LOG(INFO) << "Scaling to " << scaled_width << " x " << scaled_height
                    << " From " << img->cols << " x " << img->rows;
          */
          cv::resize(
              *img,
              scaled_img,
              cv::Size(scaled_width, scaled_height),
              0,
              0,
              cv::INTER_AREA);
          *img = scaled_img;
        }
      }

      // TODO(Yangqing): return false if any error happens.
      return true;
        */
    }

    /// assume HWC order and color channels BGR
    #[inline] pub fn saturation(
        &mut self,
        img:          *mut f32,
        img_size:     i32,
        alpha_rand:   f32,
        randgen:      *mut mt19937::MT19937) 
    {
        todo!();
        /*
            float alpha = 1.0f +
              std::uniform_real_distribution<float>(-alpha_rand, alpha_rand)(*randgen);
          // BGR to Gray scale image: R -> 0.299, G -> 0.587, B -> 0.114
          int p = 0;
          for (int h = 0; h < img_size; ++h) {
            for (int w = 0; w < img_size; ++w) {
              float gray_color = img[3 * p] * 0.114f + img[3 * p + 1] * 0.587f +
                  img[3 * p + 2] * 0.299f;
              for (int c = 0; c < 3; ++c) {
                img[3 * p + c] = img[3 * p + c] * alpha + gray_color * (1.0f - alpha);
              }
              p++;
            }
          }
        */
    }

    /// assume HWC order and color channels BGR
    #[inline] pub fn brightness(
        &mut self, 
        img: *mut f32,
        img_size: i32,
        alpha_rand: f32,
        randgen: *mut mt19937::MT19937) 
    {
        todo!();
        /*
            float alpha = 1.0f +
              std::uniform_real_distribution<float>(-alpha_rand, alpha_rand)(*randgen);
          int p = 0;
          for (int h = 0; h < img_size; ++h) {
            for (int w = 0; w < img_size; ++w) {
              for (int c = 0; c < 3; ++c) {
                img[p++] *= alpha;
              }
            }
          }
        */
    }

    /// assume HWC order and color channels BGR
    #[inline] pub fn contrast(
        &mut self, 
        img: *mut f32,
        img_size: i32,
        alpha_rand: f32,
        randgen: *mut mt19937::MT19937) 
    {
        todo!();
        /*
            float gray_mean = 0;
          int p = 0;
          for (int h = 0; h < img_size; ++h) {
            for (int w = 0; w < img_size; ++w) {
              // BGR to Gray scale image: R -> 0.299, G -> 0.587, B -> 0.114
              gray_mean += img[3 * p] * 0.114f + img[3 * p + 1] * 0.587f +
                  img[3 * p + 2] * 0.299f;
              p++;
            }
          }
          gray_mean /= (img_size * img_size);

          float alpha = 1.0f +
              std::uniform_real_distribution<float>(-alpha_rand, alpha_rand)(*randgen);
          p = 0;
          for (int h = 0; h < img_size; ++h) {
            for (int w = 0; w < img_size; ++w) {
              for (int c = 0; c < 3; ++c) {
                img[p] = img[p] * alpha + gray_mean * (1.0f - alpha);
                p++;
              }
            }
          }
        */
    }

    /// assume HWC order and color channels BGR
    #[inline] pub fn color_jitter(
        &mut self, 
        img:         *mut f32,
        img_size:    i32,
        saturation:  f32,
        brightness:  f32,
        contrast:    f32,
        randgen:     *mut mt19937::MT19937) 
    {
        todo!();
        /*
            std::srand(unsigned(std::time(0)));
          std::vector<int> jitter_order{0, 1, 2};
          // obtain a time-based seed:
          unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
          std::shuffle(
              jitter_order.begin(),
              jitter_order.end(),
              std::default_random_engine(seed));

          for (int i = 0; i < 3; ++i) {
            if (jitter_order[i] == 0) {
              Saturation<Context>(img, img_size, saturation, randgen);
            } else if (jitter_order[i] == 1) {
              Brightness<Context>(img, img_size, brightness, randgen);
            } else {
              Contrast<Context>(img, img_size, contrast, randgen);
            }
          }
        */
    }

    /// assume HWC order and color channels BGR
    #[inline] pub fn color_lighting(
        &mut self, 
        img:        *mut f32,
        img_size:   i32,
        alpha_std:  f32,
        eigvecs:    &Vec<Vec<f32>>,
        eigvals:    &Vec<f32>,
        randgen:    *mut mt19937::MT19937) 
    {
        todo!();
        /*
            std::normal_distribution<float> d(0, alpha_std);
          std::vector<float> alphas(3);
          for (int i = 0; i < 3; ++i) {
            alphas[i] = d(*randgen);
          }

          std::vector<float> delta_rgb(3, 0.0);
          for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
              delta_rgb[i] += eigvecs[i][j] * eigvals[j] * alphas[j];
            }
          }

          int p = 0;
          for (int h = 0; h < img_size; ++h) {
            for (int w = 0; w < img_size; ++w) {
              for (int c = 0; c < 3; ++c) {
                img[p++] += delta_rgb[2 - c];
              }
            }
          }
        */
    }

    /**
      | assume HWC order and color channels
      | BGR mean subtraction and scaling.
      |
      */
    #[inline] pub fn color_normalization(
        &mut self, 
        img:       *mut f32,
        img_size:  i32,
        channels:  i32,
        mean:      &Vec<f32>,
        std:       &Vec<f32>) 
    {
        todo!();
        /*
            int p = 0;
          for (int h = 0; h < img_size; ++h) {
            for (int w = 0; w < img_size; ++w) {
              for (int c = 0; c < channels; ++c) {
                img[p] = (img[p] - mean[c]) * std[c];
                p++;
              }
            }
          }
        */
    }

    /**
      | Parse datum, decode image, perform
      | transform
      | 
      | Intended as entry point for binding
      | to thread pool
      |
      */
    #[inline] pub fn decode_and_transform(
        &mut self, 
        value:          &mut String,
        image_data:     *mut f32,
        item_id:        i32,
        channels:       i32,
        thread_index:   usize)
    {
        todo!();
        /*
            CAFFE_ENFORCE((int)thread_index < num_decode_threads_);

      std::bernoulli_distribution mirror_this_image(0.5f);
      std::mt19937* randgen = &(randgen_per_thread_[thread_index]);

      cv::Mat img;
      // Decode the image
      PerImageArg info;
      CHECK(
          GetImageAndLabelAndInfoFromDBValue(value, &img, info, item_id, randgen));
      // Factor out the image transformation
      TransformImage<Context>(
          img,
          channels,
          image_data,
          color_jitter_,
          img_saturation_,
          img_brightness_,
          img_contrast_,
          color_lighting_,
          color_lighting_std_,
          color_lighting_eigvecs_,
          color_lighting_eigvals_,
          crop_,
          mirror_,
          mean_,
          std_,
          randgen,
          &mirror_this_image,
          is_test_);
        */
    }
    
    #[inline] pub fn decode_and_transpose_only(
        &mut self, 
        value:        &String,
        image_data:   *mut u8,
        item_id:      i32,
        channels:     i32,
        thread_index: usize)  
    {
        todo!();
        /*
            CAFFE_ENFORCE((int)thread_index < num_decode_threads_);

      std::bernoulli_distribution mirror_this_image(0.5f);
      std::mt19937* randgen = &(randgen_per_thread_[thread_index]);

      cv::Mat img;
      // Decode the image
      PerImageArg info;
      CHECK(
          GetImageAndLabelAndInfoFromDBValue(value, &img, info, item_id, randgen));

      // Factor out the image transformation
      CropTransposeImage<Context>(
          img,
          channels,
          image_data,
          crop_,
          mirror_,
          randgen,
          &mirror_this_image,
          is_test_);
        */
    }
    
    #[inline] pub fn prefetch(&mut self) -> bool {
        
        todo!();
        /*
            if (!owned_reader_.get()) {
        // if we are not owning the reader, we will get the reader pointer from
        // input. Otherwise the constructor should have already set the reader
        // pointer.
        reader_ = &OperatorStorage::Input<db::DBReader>(0);
      }
      const int channels = color_ ? 3 : 1;
      // Call mutable_data() once to allocate the underlying memory.
      if (gpu_transform_) {
        // we'll transfer up in int8, then convert later
        prefetched_image_.mutable_data<uint8_t>();
      } else {
        prefetched_image_.mutable_data<float>();
      }

      prefetched_label_.mutable_data<int>();
      // Prefetching handled with a thread pool of "decode_threads" threads.

      for (int item_id = 0; item_id < batch_size_; ++item_id) {
        std::string key, value;
        cv::Mat img;

        // read data
        reader_->Read(&key, &value);

        // determine label type based on first item
        if (item_id == 0) {
          if (use_caffe_datum_) {
            prefetched_label_.mutable_data<int>();
          } else {
            TensorProtos protos;
            CAFFE_ENFORCE(protos.ParseFromString(value));
            TensorProto_DataType labeldt = protos.protos(1).data_type();
            if (labeldt == TensorProto::INT32) {
              prefetched_label_.mutable_data<int>();
            } else if (labeldt == TensorProto::FLOAT) {
              prefetched_label_.mutable_data<float>();
            } else {
              LOG(FATAL) << "Unsupported label type.";
            }

            for (int i = 0; i < additional_inputs_count_; ++i) {
              int index = additional_inputs_offset_ + i;
              TensorProto additional_output_proto = protos.protos(index);
              auto sizes =
                  std::vector<int64_t>({batch_size_, additional_output_sizes_[i]});
              if (additional_output_proto.data_type() == TensorProto::FLOAT) {
                prefetched_additional_outputs_[i] =
                    caffe2::empty(sizes, at::dtype<float>().device(CPU));
              } else if (
                  additional_output_proto.data_type() == TensorProto::INT32) {
                prefetched_additional_outputs_[i] =
                    caffe2::empty(sizes, at::dtype<int>().device(CPU));
              } else if (
                  additional_output_proto.data_type() == TensorProto::INT64) {
                prefetched_additional_outputs_[i] =
                    caffe2::empty(sizes, at::dtype<int64_t>().device(CPU));
              } else if (
                  additional_output_proto.data_type() == TensorProto::UINT8) {
                prefetched_additional_outputs_[i] =
                    caffe2::empty(sizes, at::dtype<uint8_t>().device(CPU));
              } else {
                LOG(FATAL) << "Unsupported output type.";
              }
            }
          }
        }

        // launch into thread pool for processing
        // TODO: support color jitter and color lighting in gpu_transform
        if (gpu_transform_) {
          // output of decode will still be int8
          uint8_t* image_data = prefetched_image_.mutable_data<uint8_t>() +
              crop_ * crop_ * channels * item_id;
          thread_pool_->runTaskWithID(std::bind(
              &ImageInputOp<Context>::DecodeAndTransposeOnly,
              this,
              std::string(value),
              image_data,
              item_id,
              channels,
              std::placeholders::_1));
        } else {
          float* image_data = prefetched_image_.mutable_data<float>() +
              crop_ * crop_ * channels * item_id;
          thread_pool_->runTaskWithID(std::bind(
              &ImageInputOp<Context>::DecodeAndTransform,
              this,
              std::string(value),
              image_data,
              item_id,
              channels,
              std::placeholders::_1));
        }
      }
      thread_pool_->waitWorkComplete();

      // we allow to get at most max_decode_error_ratio from
      // opencv imdecode until raising a runtime exception
      if ((float)num_decode_errors_in_batch_ / batch_size_ >
          max_decode_error_ratio_) {
        throw std::runtime_error(
            "max_decode_error_ratio exceeded " +
            c10::to_string(max_decode_error_ratio_));
      }

      // If the context is not CPUContext, we will need to do a copy in the
      // prefetch function as well.
      auto device = at::device(Context::GetDeviceType());
      if (!std::is_same<Context, CPUContext>::value) {
        // do sync copies
        ReinitializeAndCopyFrom(
            &prefetched_image_on_device_, device, prefetched_image_);
        ReinitializeAndCopyFrom(
            &prefetched_label_on_device_, device, prefetched_label_);

        for (int i = 0; i < prefetched_additional_outputs_on_device_.size(); ++i) {
          ReinitializeAndCopyFrom(
              &prefetched_additional_outputs_on_device_[i],
              device,
              prefetched_additional_outputs_[i]);
        }
      }

      num_decode_errors_in_batch_ = 0;

      return true;
        */
    }

    #[inline] pub fn copy_prefetched(&mut self) -> bool {
        
        todo!();
        /*
            auto type = Device(Context::GetDeviceType());
      auto options = at::device(type);

      // Note(jiayq): The if statement below should be optimized away by the
      // compiler since std::is_same is a constexpr.
      if (std::is_same<Context, CPUContext>::value) {
        OperatorStorage::OutputTensorCopyFrom(
            0, options, prefetched_image_, /* async */ true);
        OperatorStorage::OutputTensorCopyFrom(
            1, options, prefetched_label_, /* async */ true);

        for (int i = 2; i < OutputSize(); ++i) {
          OperatorStorage::OutputTensorCopyFrom(
              i, options, prefetched_additional_outputs_[i - 2], /* async */ true);
        }
      } else {
        // TODO: support color jitter and color lighting in gpu_transform
        if (gpu_transform_) {
          if (!mean_std_copied_) {
            ReinitializeTensor(
                &mean_gpu_,
                {static_cast<int64_t>(mean_.size())},
                at::dtype<float>().device(Context::GetDeviceType()));
            ReinitializeTensor(
                &std_gpu_,
                {static_cast<int64_t>(std_.size())},
                at::dtype<float>().device(Context::GetDeviceType()));

            context_.template CopyFromCPU<float>(
                mean_.size(),
                mean_.data(),
                mean_gpu_.template mutable_data<float>());
            context_.template CopyFromCPU<float>(
                std_.size(), std_.data(), std_gpu_.template mutable_data<float>());
            mean_std_copied_ = true;
          }
          const auto& X = prefetched_image_on_device_;
          // data comes in as NHWC
          const int N = X.dim32(0), C = X.dim32(3), H = X.dim32(1), W = X.dim32(2);
          // data goes out as NCHW
          auto dims = std::vector<int64_t>{N, C, H, W};
          if (!ApplyTransformOnGPU(dims, type)) {
            return false;
          }

        } else {
          OperatorStorage::OutputTensorCopyFrom(
              0, type, prefetched_image_on_device_, /* async */ true);
        }
        OperatorStorage::OutputTensorCopyFrom(
            1, type, prefetched_label_on_device_, /* async */ true);

        for (int i = 2; i < OutputSize(); ++i) {
          OperatorStorage::OutputTensorCopyFrom(
              i,
              type,
              prefetched_additional_outputs_on_device_[i - 2],
              /* async */ true);
        }
      }
      return true;
        */
    }
}
