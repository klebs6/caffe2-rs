crate::ix!();

#[USE_OPERATOR_FUNCTIONS(Context)]
pub struct ByteWeightDequantOp<Context> {

    storage: OperatorStorage,
    context: Context,

    min:     f32,
    max:     f32,
    shape:   Vec<i64>,
}

impl<Context> ByteWeightDequantOp<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            min_(this->template GetSingleArgument<float>("min", -3)),
            max_(this->template GetSingleArgument<float>("max", 3)),
            shape_(this->template GetRepeatedArgument<int64_t>("shape"))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& WI = Input(0);

        auto* Y = Output(0, shape_, at::dtype<float>());
        float bin_interval = (max_ - min_) / 255.0;
        int total = 1;
        for (auto i = 0U; i < shape_.size(); i++) {
          total *= Y->size(i);
        }
        const uint8_t* Xdata;
        if (WI.template IsType<uint8_t>()) {
          CAFFE_ENFORCE(total, WI.nbytes());
          Xdata = WI.template data<uint8_t>();
        } else {
          CAFFE_ENFORCE(total, WI.template data<std::string>()[0].size());
          Xdata = reinterpret_cast<const uint8_t*>(
              WI.template data<std::string>()[0].c_str());
        }
        auto* Ydata = Y->template mutable_data<float>();
        ConstEigenVectorMap<uint8_t> index(&Xdata[0], total);
        EigenVectorMap<float> weights(&Ydata[0], total);
        weights = (index.cast<float>().array() * bin_interval) + min_;
        return true;
        */
    }
}

register_cpu_operator!{
    ByteWeightDequant, 
    ByteWeightDequantOp<CPUContext>
}

num_inputs!{ByteWeightDequant, 1}

num_outputs!{ByteWeightDequant, 1}
