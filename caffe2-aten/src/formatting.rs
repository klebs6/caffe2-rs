crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/Formatting.cpp]

impl fmt::Display for &mut Backend {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            return out << toString(b);
        */
    }
}

/// not all C++ compilers have default float so we
/// define our own here
///
#[inline] pub fn defaultfloat(base: &mut IosBase) -> &mut IosBase {
    
    todo!();
        /*
            __base.unsetf(ios_base::floatfield);
      return __base;
        */
}

/**
  | saves/restores number formatting
  | inside scope
  |
  */
pub struct FormatGuard {
    out:   &mut std::io::BufWriter,
    saved: ios,
}

impl FormatGuard {
    
    pub fn new(out: &mut std::io::BufWriter) -> Self {
    
        todo!();
        /*
        : out(out),
        : saved(nullptr),

            saved.copyfmt(out);
        */
    }
}

impl Drop for FormatGuard {
    fn drop(&mut self) {
        todo!();
        /*
            out.copyfmt(saved);
        */
    }
}

impl fmt::Display for &mut DeprecatedTypeProperties {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            return out << t.toString();
        */
    }
}

pub fn print_format(
        stream: &mut std::io::BufWriter,
        self_:  &Tensor) -> (f64,i64) {
    
    todo!();
        /*
            auto size = self.numel();
      if(size == 0) {
        return make_tuple(1., 0);
      }
      bool intMode = true;
      auto self_p = self.data_ptr<double>();
      for(i64 i = 0; i < size; i++) {
        auto z = self_p[i];
        if(isfinite(z)) {
          if(z != ceil(z)) {
            intMode = false;
            break;
          }
        }
      }
      i64 offset = 0;
      while(!isfinite(self_p[offset])) {
        offset = offset + 1;
        if(offset == size) {
          break;
        }
      }
      double expMin;
      double expMax;
      if(offset == size) {
        expMin = 1;
        expMax = 1;
      } else {
        expMin = fabs(self_p[offset]);
        expMax = fabs(self_p[offset]);
        for(i64 i = offset; i < size; i++) {
          double z = fabs(self_p[i]);
          if(isfinite(z)) {
            if(z < expMin) {
              expMin = z;
            }
            if(self_p[i] > expMax) {
              expMax = z;
            }
          }
        }
        if(expMin != 0) {
          expMin = floor(log10(expMin)) + 1;
        } else {
          expMin = 1;
        }
        if(expMax != 0) {
          expMax = floor(log10(expMax)) + 1;
        } else {
          expMax = 1;
        }
      }
      double scale = 1;
      i64 sz;
      if(intMode) {
        if(expMax > 9) {
          sz = 11;
          stream << scientific << setprecision(4);
        } else {
          sz = expMax + 1;
          stream << defaultfloat;
        }
      } else {
        if(expMax-expMin > 4) {
          sz = 11;
          if(fabs(expMax) > 99 || fabs(expMin) > 99) {
            sz = sz + 1;
          }
          stream << scientific << setprecision(4);
        } else {
          if(expMax > 5 || expMax < 0) {
            sz = 7;
            scale = pow(10, expMax-1);
            stream << fixed << setprecision(4);
          } else {
            if(expMax == 0) {
              sz = 7;
            } else {
              sz = expMax+6;
            }
            stream << fixed << setprecision(4);
          }
        }
      }
      return make_tuple(scale, sz);
        */
}

pub fn print_indent(
        stream: &mut std::io::BufWriter,
        indent: i64)  {
    
    todo!();
        /*
            for(i64 i = 0; i < indent; i++) {
        stream << " ";
      }
        */
}

pub fn print_scale(
        stream: &mut std::io::BufWriter,
        scale:  f64)  {
    
    todo!();
        /*
            FormatGuard guard(stream);
      stream << defaultfloat << scale << " *" << endl;
        */
}

pub fn print_matrix(
        stream:   &mut std::io::BufWriter,
        self_:    &Tensor,
        linesize: i64,
        indent:   i64)  {
    
    todo!();
        /*
      double scale;
      i64 sz;
      tie(scale, sz) = __printFormat(stream, self);

      __printIndent(stream, indent);
      i64 nColumnPerLine = (linesize-indent)/(sz+1);
      i64 firstColumn = 0;
      i64 lastColumn = -1;
      while(firstColumn < self.size(1)) {
        if(firstColumn + nColumnPerLine <= self.size(1)) {
          lastColumn = firstColumn + nColumnPerLine - 1;
        } else {
          lastColumn = self.size(1) - 1;
        }
        if(nColumnPerLine < self.size(1)) {
          if(firstColumn != 0) {
            stream << endl;
          }
          stream << "Columns " << firstColumn+1 << " to " << lastColumn+1;
          __printIndent(stream, indent);
        }
        if(scale != 1) {
          printScale(stream,scale);
          __printIndent(stream, indent);
        }
        for(i64 l = 0; l < self.size(0); l++) {
          Tensor row = self.select(0,l);
          double *row_ptr = row.data_ptr<double>();
          for(i64 c = firstColumn; c < lastColumn+1; c++) {
            stream << setw(sz) << row_ptr[c]/scale;
            if(c == lastColumn) {
              stream << endl;
              if(l != self.size(0)-1) {
                if(scale != 1) {
                  __printIndent(stream, indent);
                  stream << " ";
                } else {
                  __printIndent(stream, indent);
                }
              }
            } else {
              stream << " ";
            }
          }
        }
        firstColumn = lastColumn + 1;
      }
        */
}

pub fn print_tensor(
        stream:   &mut std::io::BufWriter,
        self_:    &mut Tensor,
        linesize: i64)  {
    
    todo!();
        /*
            vector<i64> counter(self.ndimension()-2);
      bool start = true;
      bool finished = false;
      counter[0] = -1;
      for(usize i = 1; i < counter.size(); i++)
        counter[i] = 0;
      while(true) {
        for(i64 i = 0; self.ndimension()-2; i++) {
          counter[i] = counter[i] + 1;
          if(counter[i] >= self.size(i)) {
            if(i == self.ndimension()-3) {
              finished = true;
              break;
            }
            counter[i] = 0;
          } else {
            break;
          }
        }
        if(finished) {
          break;
        }
        if(start) {
          start = false;
        } else {
          stream << endl;
        }
        stream << "(";
        Tensor tensor = self;
        for(i64 i=0; i < self.ndimension()-2; i++) {
          tensor = tensor.select(0, counter[i]);
          stream << counter[i]+1 << ",";
        }
        stream << ".,.) = " << endl;
        __printMatrix(stream, tensor, linesize, 1);
      }
        */
}

pub fn print(
        stream:   &mut std::io::BufWriter,
        tensor:   &Tensor,
        linesize: i64) -> &mut std::io::BufWriter {
    
    todo!();
        /*
            FormatGuard guard(stream);
      if(!tensor_.defined()) {
        stream << "[ Tensor (undefined) ]";
      } else if (tensor_.is_sparse()) {
        stream << "[ " << tensor_.toString() << "{}\n";
        stream << "indices:\n" << tensor_._indices() << "\n";
        stream << "values:\n" << tensor_._values() << "\n";
        stream << "size:\n" << tensor_.sizes() << "\n";
        stream << "]";
      } else {
        Tensor tensor;
        if (tensor_.is_quantized()) {
          tensor = tensor_.dequantize().to(kCPU, kDouble).contiguous();
        } else if (tensor_.is_mkldnn()) {
          stream << "MKLDNN Tensor: ";
          tensor = tensor_.to_dense().to(kCPU, kDouble).contiguous();
        } else {
          tensor = tensor_.to(kCPU, kDouble).contiguous();
        }
        if(tensor.ndimension() == 0) {
          stream << defaultfloat << tensor.data_ptr<double>()[0] << endl;
          stream << "[ " << tensor_.toString() << "{}";
        } else if(tensor.ndimension() == 1) {
          if (tensor.numel() > 0) {
            double scale;
            i64 sz;
            tie(scale, sz) =  __printFormat(stream, tensor);
            if(scale != 1) {
              printScale(stream, scale);
            }
            double* tensor_p = tensor.data_ptr<double>();
            for (i64 i = 0; i < tensor.size(0); i++) {
              stream << setw(sz) << tensor_p[i]/scale << endl;
            }
          }
          stream << "[ " << tensor_.toString() << "{" << tensor.size(0) << "}";
        } else if(tensor.ndimension() == 2) {
          if (tensor.numel() > 0) {
            __printMatrix(stream, tensor, linesize, 0);
          }
          stream << "[ " << tensor_.toString() << "{" << tensor.size(0) << "," <<  tensor.size(1) << "}";
        } else {
          if (tensor.numel() > 0) {
            __printTensor(stream, tensor, linesize);
          }
          stream << "[ " << tensor_.toString() << "{" << tensor.size(0);
          for(i64 i = 1; i < tensor.ndimension(); i++) {
            stream << "," << tensor.size(i);
          }
          stream << "}";
        }
        if (tensor_.is_quantized()) {
          stream << ", qscheme: " << toString(tensor_.qscheme());
          if (tensor_.qscheme() == kPerTensorAffine) {
            stream << ", scale: " << tensor_.q_scale();
            stream << ", zero_point: " << tensor_.q_zero_point();
          } else if (tensor_.qscheme() == kPerChannelAffine ||
              tensor_.qscheme() == kPerChannelAffineFloatQParams) {
            stream << ", scales: ";
            Tensor scales = tensor_.q_per_channel_scales();
            print(stream, scales, linesize);
            stream << ", zero_points: ";
            Tensor zero_points = tensor_.q_per_channel_zero_points();
            print(stream, zero_points, linesize);
            stream << ", axis: " << tensor_.q_per_channel_axis();
          }
        }

        // Proxy check for if autograd was built
        if (tensor.getIntrusivePtr()->autograd_meta()) {
          auto& fw_grad = tensor._fw_grad(/* level */ 0);
          if (fw_grad.defined()) {
            stream << ", tangent:" << endl << fw_grad;
          }
        }
        stream << " ]";
      }
      return stream;
        */
}
