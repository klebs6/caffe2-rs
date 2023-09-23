crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/vmap_test.cpp]

#[test] fn vmap_test_batched_tensor() {
    todo!();
    /*
    
      {
        Tensor x = addBatchDim(ones({2, 3, 4}), /*lvl=*/1, /*dim=*/1);
        vector<i64> expected_size = {2, 4};
        ASSERT_EQ(x.sizes(), expected_size);
        ASSERT_EQ(x.dim(), 2);
        ASSERT_EQ(x.numel(), 8);
        ASSERT_EQ(x.is_contiguous(), false);
        ASSERT_THROW(x.storage(), Error);
        ASSERT_EQ(x.storage_offset(), 0);
      }
      {
        // Test multiple batch dims
        Tensor x = addBatchDim(ones({2, 3, 4}), /*lvl=*/1, /*dim=*/1);
        x = addBatchDim(x, /*lvl=*/2, /*dim=*/1);
        vector<i64> expected_size = {2};
        ASSERT_EQ(x.sizes(), expected_size);
        ASSERT_EQ(x.dim(), 1);
        ASSERT_EQ(x.numel(), 2);
      }
      {
        // Test vmap tensor dimensionality limit

        // Should not throw
        vector<i64> sizes(kVmapMaxTensorDims, 1);
        Tensor x = addBatchDim(ones(sizes), /*lvl=*/1, /*dim=*/1);

        // Should throw
        vector<i64> too_many_sizes(kVmapMaxTensorDims + 1, 1);
        auto big_dim_tensor = ones(too_many_sizes);
        ASSERT_THROW(addBatchDim(big_dim_tensor, /*lvl=*/1, /*dim=*/1), Error);
      }
      {
        // Create a "scalar" BatchedTensor. Should not crash.
        Tensor tensor = addBatchDim(ones({3}), /*lvl*/1, /*dim*/0);
      }

    */
}

// returns {{lvl=0,dim=0}, {lvl=1,dim=1}, ..., {lvl=kVmapNumLevels-1,dim=kVmapNumLevels-1}};
//
pub fn max_batch_dims_at_front() -> BatchDims {
    
    todo!();
        /*
            BatchDims result;
      for (i64 lvl = 0; lvl < kVmapNumLevels; lvl++) {
        result.emplace_back(lvl, /*dim=*/lvl);
      }
      return result;
        */
}

#[test] fn vmap_test_batched_tensor_max_level() {
    todo!();
    /*
    
      {
        // Should not throw
        auto tensor = ones({2, 3, 4});
        makeBatched(ones({2, 3, 4}), {{/*lvl*/kVmapNumLevels - 1, /*dim*/0}});
      }
      {
        auto tensor = ones({2, 3, 4});
        ASSERT_THROW(
            makeBatched(ones({2, 3, 4}), {{/*lvl*/kVmapNumLevels, /*dim*/0}}),
            Error);
      }
      {
        auto tensor = ones({2, 3, 4});
        ASSERT_THROW(
            makeBatched(ones({2, 3, 4}), {{/*lvl*/kVmapNumLevels + 5, /*dim*/0}}),
            Error);
      }
      {
        // create a BatchedTensor with kVmapNumLevels levels.
        // Should not throw
        auto tensor = ones(vector<i64>(kVmapNumLevels, 1));
        makeBatched(tensor, maxBatchDimsAtFront());
      }
      {
        // create a BatchedTensor with kVmapNumLevels+1 levels.
        auto tensor = ones(vector<i64>(kVmapNumLevels + 1, 1));
        auto batch_dims = maxBatchDimsAtFront();
        batch_dims.emplace_back(/*lvl*/kVmapNumLevels, /*dim*/kVmapNumLevels);
        ASSERT_THROW(makeBatched(tensor, batch_dims), Error);
      }

    */
}

#[test] fn vmap_test_batched_tensor_actual_dim() {
    todo!();
    /*
    
      {
        // No batch dims
        Tensor tensor = makeBatched(ones({2, 3, 5, 7}), {});
        auto* batched = maybeGetBatchedImpl(tensor);
        ASSERT_EQ(batched->actualDim(0), 0);
        ASSERT_EQ(batched->actualDim(1), 1);
        ASSERT_EQ(batched->actualDim(3), 3);

        // Test wrap around
        ASSERT_EQ(batched->actualDim(-1), 3);
        ASSERT_EQ(batched->actualDim(-4), 0);
        ASSERT_THROW(batched->actualDim(-5), Error);
        ASSERT_THROW(batched->actualDim(4), Error);

        // test wrap_dim = False
        ASSERT_THROW(batched->actualDim(-1, /*wrap_dim*/false), Error);
        ASSERT_THROW(batched->actualDim(-4, /*wrap_dim*/false), Error);
      }
      {
        // Single batch dim at front
        Tensor tensor = makeBatched(ones({2, 3, 5, 7}), {{/*lvl*/1, /*dim*/0}});
        auto* batched = maybeGetBatchedImpl(tensor);
        ASSERT_EQ(batched->actualDim(0), 1);
        ASSERT_EQ(batched->actualDim(2), 3);
        ASSERT_EQ(batched->actualDim(-1), 3);
        ASSERT_THROW(batched->actualDim(3), Error);
      }
      {
        // Single batch dim in middle
        Tensor tensor = makeBatched(ones({2, 3, 5, 7}), {{/*lvl*/1, /*dim*/1}});
        auto* batched = maybeGetBatchedImpl(tensor);
        ASSERT_EQ(batched->actualDim(0), 0);
        ASSERT_EQ(batched->actualDim(1), 2);
        ASSERT_EQ(batched->actualDim(2), 3);
      }
      {
        // Single batch dim at end
        Tensor tensor = makeBatched(ones({2, 3, 5, 7}), {{/*lvl*/1, /*dim*/1}});
        auto* batched = maybeGetBatchedImpl(tensor);
        ASSERT_EQ(batched->actualDim(0), 0);
        ASSERT_EQ(batched->actualDim(2), 3);
        ASSERT_EQ(batched->actualDim(-1), 3);
      }
      {
        // Multiple (2) batch dims at front
        Tensor tensor = makeBatched(
            ones({2, 3, 5, 7}),
            {{/*lvl*/1, /*dim*/0}, {/*lvl*/2, /*dim*/1}});
        auto* batched = maybeGetBatchedImpl(tensor);
        ASSERT_EQ(batched->actualDim(0), 2);
        ASSERT_EQ(batched->actualDim(1), 3);
      }
      {
        // Multiple (2) batch dims, misc places
        Tensor tensor = makeBatched(
            ones({2, 3, 5, 7}),
            {{/*lvl*/1, /*dim*/1}, {/*lvl*/2, /*dim*/3}});
        auto* batched = maybeGetBatchedImpl(tensor);
        ASSERT_EQ(batched->actualDim(0), 0);
        ASSERT_EQ(batched->actualDim(1), 2);
        ASSERT_EQ(batched->actualDim(-1), 2);
        ASSERT_EQ(batched->actualDim(-2), 0);
      }
      {
        // ActualDim on kVmapMaxTensorDims sized underlying tensor
        auto tensor = ones({});
        for (i64 i = 0; i < kVmapMaxTensorDims; i++) {
          tensor = tensor.unsqueeze(0);
        }
        ASSERT_EQ(tensor.dim(), kVmapMaxTensorDims);

        auto batched = addBatchDim(tensor, /*lvl*/1, /*dim*/0);
        auto* batched_impl = maybeGetBatchedImpl(batched);
        ASSERT_EQ(
            batched_impl->actualDim(kVmapMaxTensorDims - 2),
            kVmapMaxTensorDims - 1);
        ASSERT_EQ(
            batched_impl->actualDim(-1),
            kVmapMaxTensorDims - 1);
      }

    */
}

#[test] fn vmap_test_multi_batch_transform() {
    todo!();
    /*
    
      {
        // Input is regular Tensor
        auto tensor = ones({2, 3, 5});
        ASSERT_THROW(MultiBatchVmapTransform::logicalToPhysical(tensor), Error);
      }
      {
        // Input is BatchedTensor, Batch dims are already at the front
        auto tensor = ones({2, 3, 5});
        BatchDims bdims = {{/*lvl*/1, /*dim*/0}, {/*lvl*/3, /*dim*/1}};
        auto batched = makeBatched(tensor, bdims);

        auto result = MultiBatchVmapTransform::logicalToPhysical(batched);
        ASSERT_TRUE(result.tensor().is_same(tensor));
      }
      {
        // Single batch dim, not at front
        auto tensor = ones({2, 3, 5});
        BatchDims bdims = {{/*lvl*/1, /*dim*/1}};
        auto batched = makeBatched(tensor, bdims);

        auto result = MultiBatchVmapTransform::logicalToPhysical(batched);
        ASSERT_EQ(result.tensor().data_ptr(), tensor.data_ptr());
        ASSERT_TRUE(allclose(result.tensor(), tensor.permute({1, 0, 2})));
      }
      {
        // Multiple batch dims, not at front.
        auto tensor = ones({2, 3, 5});
        BatchDims bdims = {{/*lvl*/1, /*dim*/1}, {/*lvl*/2,/*dim*/2}, {/*lvl*/3,/*dim*/0}};
        auto batched = makeBatched(tensor, bdims);

        auto result = MultiBatchVmapTransform::logicalToPhysical(batched);
        ASSERT_EQ(result.tensor().data_ptr(), tensor.data_ptr());
        ASSERT_TRUE(allclose(result.tensor(), tensor.permute({1, 2, 0})));
      }
      {
        // Edge case: kVmapNumLevels levels; batch dims are already at front.

        // sizes=[2, 1, 3, 1, 1, 7, 1, 1, 1, 1, ...]
        auto sizes = vector<i64>(kVmapNumLevels, 1);
        sizes[0] = 2;
        sizes[2] = 3;
        sizes[5] = 7;

        // bdims = {{lvl=0,dim=0,lvl=1,dim=1,...,{lvl=63,dim=63}}
        auto batch_dims = maxBatchDimsAtFront();
        auto tensor = ones(sizes);

        auto batched = makeBatched(tensor, batch_dims);
        auto result = MultiBatchVmapTransform::logicalToPhysical(batched);
        ASSERT_TRUE(result.tensor().is_same(tensor));
      }
      {
        // Edge case: kVmapNumLevels levels; batch dims are not at front

        // sizes=[1, 3, 2, 1, 1, 7, 1, 1, 1, 1, ..., 1, 1, 5]
        auto sizes = vector<i64>(kVmapNumLevels, 1);
        sizes[1] = 3;
        sizes[2] = 2;
        sizes[5] = 7;
        sizes[kVmapNumLevels - 1] = 5;

        // The goal is to permute sizes such that the final sizes are:
        // [2, 3, 5, 7, 1, 1, 1, 1, 1, ...]
        auto expected_result_sizes = vector<i64>(kVmapNumLevels, 1);
        expected_result_sizes[0] = 2;
        expected_result_sizes[1] = 3;
        expected_result_sizes[2] = 5;
        expected_result_sizes[3] = 7;

        // bdims = {{0, 2}, {1, 1}, {2, 63}, {3, 5}, {4, 0}, {5, 3}, {6, 4},
        //          {7, 6}, {8, 7}, {9, 8}, ..., {63, 62}}
        BatchDims batch_dims = {
          {0, 2}, {1, 1}, {2, kVmapNumLevels - 1}, {3, 5}, {4, 0}, {5, 3}, {6, 4}
        };
        for (i64 level = 7; level < kVmapNumLevels; level++ ) {
          batch_dims.emplace_back(level, /*dim=*/level - 1);
        }
        auto tensor = ones(sizes);

        auto batched = makeBatched(tensor, batch_dims);
        auto result = MultiBatchVmapTransform::logicalToPhysical(batched);
        ASSERT_EQ(result.tensor().data_ptr(), tensor.data_ptr());
        ASSERT_EQ(result.tensor().sizes(), expected_result_sizes);
      }

    */
}

#[test] fn vmap_test_physical_view_get_dim() {
    todo!();
    /*
    
      VmapPhysicalView physical_view(ones({2, 3, 4, 5, 6}), 1 | 4);

      // Positive dims
      ASSERT_EQ(physical_view.getPhysicalDim(0), 2);
      ASSERT_EQ(physical_view.getPhysicalDim(1), 3);
      ASSERT_EQ(physical_view.getPhysicalDim(2), 4);
      ASSERT_THROW(physical_view.getPhysicalDim(3), Error);

      // Negative dims (testing wrap dim behavior)
      ASSERT_EQ(physical_view.getPhysicalDim(-1), 4);
      ASSERT_EQ(physical_view.getPhysicalDim(-2), 3);
      ASSERT_EQ(physical_view.getPhysicalDim(-3), 2);
      ASSERT_THROW(physical_view.getPhysicalDim(-4), Error);

    */
}

#[test] fn vmap_test_physical_view_get_dims() {
    todo!();
    /*
    
      VmapPhysicalView physical_view(ones({2, 3, 4, 5, 6}), 2 | 8 | 16);

      ASSERT_EQ(
          physical_view.getPhysicalDims({0, 1, -1, -2}),
          VmapDimVector({3, 4, 4, 3}));

      ASSERT_THROW(physical_view.getPhysicalDims({2, 0}), Error);
      ASSERT_THROW(physical_view.getPhysicalDims({0, -3}), Error);

    */
}

pub fn check_batch_dims_equal(
        bdims:          BatchDimsRef,
        expected_bdims: BatchDimsRef)  {
    
    todo!();
        /*
            ASSERT_EQ(bdims.size(), expected_bdims.size());
      for (i64 idx = 0; idx < bdims.size(); idx++) {
        ASSERT_EQ(bdims[idx].dim(), expected_bdims[idx].dim());
        ASSERT_EQ(bdims[idx].level(), expected_bdims[idx].level());
      }
        */
}

#[test] fn vmap_test_physical_view_new_logical_from() {
    todo!();
    /*
    
      {
        // Simple case: single level
        VmapPhysicalView physical_view(ones({2, 3, 4}), /*levels = {2}*/4);
        Tensor physical = ones({2, 6, 7});

        auto result = physical_view.getPhysicalToLogicalMap().apply(physical);
        auto* batched = maybeGetBatchedImpl(result);
        ASSERT_TRUE(batched != nullptr);
        ASSERT_TRUE(batched->value().is_same(physical));
        checkBatchDimsEqual(batched->bdims(), {{2, 0}});
      }
      {
        // Multiple levels
        VmapPhysicalView physical_view(ones({2, 3, 4, 5, 6}), /*levels = {1, 3, 4}*/2 | 8 | 16);
        Tensor physical = ones({2, 3, 4, 7});

        auto result = physical_view.getPhysicalToLogicalMap().apply(physical);
        auto* batched = maybeGetBatchedImpl(result);
        ASSERT_TRUE(batched != nullptr);
        ASSERT_TRUE(batched->value().is_same(physical));
        checkBatchDimsEqual(batched->bdims(), {{1, 0}, {3, 1}, {4, 2}});
      }
      {
        // Logical dimensions is [].
        VmapPhysicalView physical_view(ones({2}), /*levels = {2}*/4);
        Tensor physical = ones({2});

        auto result = physical_view.getPhysicalToLogicalMap().apply(physical);
        auto* batched = maybeGetBatchedImpl(result);
        ASSERT_TRUE(batched != nullptr);
        ASSERT_TRUE(batched->value().is_same(physical));
        checkBatchDimsEqual(batched->bdims(), {{2, 0}});
      }

    */
}

/**
  | Basic test for BatchedTensor::sum.
  |
  | NB: We don't need to write tests in C++ for
  | batching rules if we can test them in Python
  | via the vmap API. These are here to bootstrap
  | that process.
  |
  */
#[test] fn vmap_test_batched_tensor_sum() {
    todo!();
    /*
    
      {
        // Simple: single batch dim, single reduce dim
        Tensor x = randn({2, 3, 5, 7});

        Tensor batched_x = makeBatched(x, {{/*lvl*/1, /*dim*/0}});
        Tensor batched_out = batched_x.sum(0);
        const auto& out = maybeGetBatchedImpl(batched_out)->value();

        ASSERT_TRUE(allclose(out, x.sum(1)));
      }
      {
        // single batch dim, -1 reduce dim handling
        Tensor x = randn({2, 3});

        Tensor batched_x = makeBatched(x, {{/*lvl*/1, /*dim*/1}});
        Tensor batched_out = batched_x.sum(-1);
        const auto& out = maybeGetBatchedImpl(batched_out)->value();

        ASSERT_TRUE(allclose(out, x.sum(0)));
      }
      {
        // single batch dim, multiple reduce dim
        Tensor x = randn({2, 3, 5, 7});

        Tensor batched_x = makeBatched(x, {{/*lvl*/1, /*dim*/1}});
        Tensor batched_out = batched_x.sum(vector<i64>{0, 1});
        const auto& out = maybeGetBatchedImpl(batched_out)->value();

        ASSERT_TRUE(allclose(out, x.sum(vector<i64>{0, 2})));
      }
      {
        // multiple batch dim, multiple reduce dim
        Tensor x = randn({2, 3, 5, 7});

        Tensor batched_x = makeBatched(x, {{/*lvl*/1, /*dim*/0}, {/*lvl*/2, /*dim*/1}});
        Tensor batched_out = batched_x.sum(vector<i64>{0, 1});
        const auto& out = maybeGetBatchedImpl(batched_out)->value();

        ASSERT_TRUE(allclose(out, x.sum(vector<i64>{2, 3})));
      }

    */
}

pub fn check_broadcasting_vmap_transform(
    inputs:           &[Tensor],
    expected_outputs: &[Tensor])  {
    
    todo!();
        /*
            auto outputs = BroadcastingVmapTransform::logicalToPhysical(inputs);
      ASSERT_EQ(outputs.size(), expected_outputs.size());
      for (i64 idx = 0; idx < outputs.size(); idx++) {
        const auto& output = outputs[idx].tensor();
        ASSERT_EQ(output.data_ptr(), expected_outputs[idx].data_ptr());
        ASSERT_TRUE(allclose(output, expected_outputs[idx]));
      }
        */
}

#[test] fn vmap_test_broadcasting_transform_batched() {
    todo!();
    /*
    
      {
        // Check that batch dims get moved to the front
        i64 B0 = 5, B1 = 7;
        Tensor x = randn({2, B0, 3, B1});
        Tensor y = randn({B1, 2, 3, B0});
        Tensor batched_x = makeBatched(x, {{0, 1}, {1, 3}});
        Tensor batched_y = makeBatched(y, {{0, 3}, {1, 0}});

        checkBroadcastingVmapTransform(
            {batched_x, batched_y},
            {x.permute({1, 3, 0, 2}), y.permute({3, 0, 1, 2})});
      }
      {
        // Check that batch dims become aligned (i.e. extra 1 dims get added)
        i64 B0 = 5, B1 = 7, B2 = 9;
        Tensor x = randn({B0, B2, 2, 3});
        Tensor y = randn({B0, B1, 2, 3});
        Tensor batched_x = makeBatched(x, {{0, 0}, {2, 1}});
        Tensor batched_y = makeBatched(y, {{0, 0}, {1, 1}});

        checkBroadcastingVmapTransform(
            {batched_x, batched_y},
            {x.unsqueeze(1), y.unsqueeze(2)});
      }
      {
        // Check that the "example" gets padded with extra dims of size 1.
        i64 B0 = 5;
        Tensor x = randn({B0, 3});
        Tensor y = randn({B0, 2, 3});
        Tensor batched_x = makeBatched(x, {{0, 0}});
        Tensor batched_y = makeBatched(y, {{0, 0}});

        checkBroadcastingVmapTransform(
            {batched_x, batched_y},
            {x.unsqueeze(1), y});
      }
      {
        // Check batch dims get moved to front, batch dims get aligned,
        // and the example gets padded correctly.
        i64 B0 = 5, B1 = 7, B2 = 11, B3 = 13;
        Tensor x = randn({2, B0, 3, B2});
        Tensor y = randn({B3, 3, B1});
        Tensor batched_x = makeBatched(x, {{0, 1}, {2, 3}});
        Tensor batched_y = makeBatched(y, {{1, 2}, {3, 0}});

        checkBroadcastingVmapTransform(
            {batched_x, batched_y},
            {
              x.permute({1, 3, 0, 2}).view({B0, 1, B2, 1, 2, 3}),
              y.permute({2, 0, 1}).view({1, B1, 1, B3, 1, 3}),
            });
      }
      {
        // Edge case: BatchedTensor "scalar" handling
        i64 B0 = 5, B2 = 11;
        Tensor x = randn({B0});
        Tensor y = randn({B0, B2});
        Tensor batched_x = makeBatched(x, {{0, 0}});
        Tensor batched_y = makeBatched(y, {{0, 0}, {1, 1}});

        checkBroadcastingVmapTransform({batched_x, batched_y}, {x.view({B0, 1}), y});
        checkBroadcastingVmapTransform({batched_y, batched_x}, {y, x.view({B0, 1})});
      }
      {
        // Edge case: Only one tensor is a "batchedtensor scalar"
        i64 B0 = 5, B2 = 11;
        Tensor x = randn({B0});
        Tensor y = randn({B0, B2, 2});
        Tensor batched_x = makeBatched(x, {{0, 0}});
        Tensor batched_y = makeBatched(y, {{0, 0}, {1, 1}});

        checkBroadcastingVmapTransform({batched_x, batched_y}, {x.view({B0, 1, 1}), y});
        checkBroadcastingVmapTransform({batched_y, batched_x}, {y, x.view({B0, 1, 1})});
      }

    */
}

#[test] fn vmap_test_broadcasting_transform_batched_unbatched() {
    todo!();
    /*
    
      {
        // Check same example size
        i64 B0 = 5, B1 = 7;
        Tensor x = randn({2, B0, 3, B1});
        Tensor y = randn({2, 3});
        Tensor batched_x = makeBatched(x, {{0, 1}, {1, 3}});

        checkBroadcastingVmapTransform(
            {batched_x, y},
            {x.permute({1, 3, 0, 2}), y.view({1, 1, 2, 3})});
        checkBroadcastingVmapTransform(
            {y, batched_x},
            {y.view({1, 1, 2, 3}), x.permute({1, 3, 0, 2})});
      }
      {
        // BatchedTensor has higher example dim than non-batched-tensor
        i64 B0 = 5, B1 = 7;
        Tensor x = randn({B0, B1, 2, 3});
        Tensor y = randn({3});
        Tensor batched_x = makeBatched(x, {{0, 0}, {1, 1}});

        checkBroadcastingVmapTransform(
            {batched_x, y}, {x, y.view({1, 1, 1, 3})});
        checkBroadcastingVmapTransform(
            {y, batched_x}, {y.view({1, 1, 1, 3}), x});
      }
      {
        // BatchedTensor has lower example dim than non-batched-tensor
        i64 B0 = 5, B1 = 7;
        Tensor x = randn({B0, B1, 3});
        Tensor y = randn({2, 3});
        Tensor batched_x = makeBatched(x, {{0, 0}, {1, 1}});

        checkBroadcastingVmapTransform(
            {batched_x, y}, {x.view({B0, B1, 1, 3}), y.view({1, 1, 2, 3})});
        checkBroadcastingVmapTransform(
            {y, batched_x}, {y.view({1, 1, 2, 3}), x.view({B0, B1, 1, 3})});
      }
      {
        // Scalar handling
        i64 B0 = 5, B1 = 7;
        Tensor x = randn({B0, B1});
        Tensor y = randn({});
        Tensor batched_x = makeBatched(x, {{0, 0}, {1, 1}});

        checkBroadcastingVmapTransform({batched_x, y}, {x, y.view({1, 1})});
        checkBroadcastingVmapTransform({y, batched_x}, {y.view({1, 1}), x});
      }

    */
}

#[test] fn vmap_test_broadcasting_transform_max_levels() {
    todo!();
    /*
    
      {
        // inputs have all 64 levels
        auto x = randn(vector<i64>(kVmapNumLevels, 1));
        auto y = randn(vector<i64>(kVmapNumLevels, 1));
        auto batched_x = makeBatched(x, maxBatchDimsAtFront());
        auto batched_y = makeBatched(y, maxBatchDimsAtFront());

        checkBroadcastingVmapTransform({batched_x, batched_y}, {x, y});
      }
      {
        // inputs don't have all 64 levels, but results do.
        i64 split = 19;
        auto x = randn(vector<i64>(split, 1));
        auto y = randn(vector<i64>(kVmapNumLevels - split, 1));

        auto tmp = maxBatchDimsAtFront();
        BatchDims x_bdims(tmp.begin(), tmp.begin() + split);

        // Construct y_bdims.
        i64 dim = 0;
        auto y_bdims_vector = fmap(
            ArrayRef<BatchDim>(tmp.begin() + split, tmp.end()),
            [&](const BatchDim& bdim) -> BatchDim {
              return { bdim.level(), dim++ };
            });
        BatchDims y_bdims(y_bdims_vector.begin(), y_bdims_vector.end());

        auto batched_x = makeBatched(x, x_bdims);
        auto batched_y = makeBatched(y, y_bdims);

        auto expected_size = vector<i64>(kVmapNumLevels, 1);
        checkBroadcastingVmapTransform(
            {batched_x, batched_y},
            {x.view(expected_size), y.view(expected_size)});
      }

    */
}

// Basic test for BatchedTensor::mul.
#[test] fn vmap_test_batched_tensor_mul() {
    todo!();
    /*
    
      {
        // batched * batched
        Tensor x = randn({2, 3});
        Tensor y = randn({2, 3});

        Tensor Bx = addBatchDim(x, /*lvl*/1, /*dim*/0);
        Tensor By = addBatchDim(y, /*lvl*/1, /*dim*/0);
        Tensor Bout = Bx * By;

        const auto& out = maybeGetBatchedImpl(Bout)->value();
        vector<i64> expected_size = {2, 3};
        ASSERT_EQ(out.sizes(), expected_size);
        ASSERT_TRUE(allclose(out, x * y));
      }
      {
        // batched * unbatched
        Tensor x = randn({2, 3});
        Tensor y = randn({3});

        Tensor Bx = addBatchDim(x, /*lvl*/1, /*dim*/0);
        Tensor Bout = Bx * y;
        const auto& out = maybeGetBatchedImpl(Bout)->value();
        vector<i64> expected_size = {2, 3};
        ASSERT_EQ(out.sizes(), expected_size);
        ASSERT_TRUE(allclose(out, x * y));
      }
      {
        // batched (level 1) * batched (level 2)
        Tensor x = randn({2, 3});
        Tensor y = randn({5, 3});

        Tensor Bx = addBatchDim(x, /*lvl*/1, /*dim*/0);
        Tensor By = addBatchDim(y, /*lvl*/2, /*dim*/0);
        Tensor Bout = Bx * By;

        // We get a doubly wrapped BatchTensor...
        const auto& out = maybeGetBatchedImpl(Bout)->value();
        vector<i64> expected_size = {2, 5, 3};
        ASSERT_EQ(out.sizes(), expected_size);
        ASSERT_TRUE(allclose(out, x.unsqueeze(1) * y));
      }
      {
        // batched (level 2, 3, 4) * batched (level 3, 1, 2)
        Tensor x = randn({3, 5, 7});
        Tensor y = randn({5, 2, 3});

        // Each BatchDim is constructed in {dim, level} format.
        Tensor Bx = makeBatched(x, {{2, 0}, {3, 1}, {4, 2}});
        Tensor By = makeBatched(y, {{1, 1}, {2, 2}, {3, 0}});
        Tensor Bout = Bx * By;

        const auto& out = maybeGetBatchedImpl(Bout)->value();

        // The batching rule aligns dimensions in the order of their `level`.
        // It just happened that we chose sizes to be in the same order as the level.
        vector<i64> expected_size = {2, 3, 5, 7};
        ASSERT_EQ(out.sizes(), expected_size);
        ASSERT_TRUE(allclose(out, x * y.permute({1, 2, 0}).unsqueeze(3)));
      }

    */
}

// test for BatchedTensor::size(int).
#[test] fn vmap_test_batched_tensor_size() {
    todo!();
    /*
    
      {
        // Single batch dim at front
        Tensor x = randn({3, 5, 7});
        Tensor Bx = makeBatched(x, {{0, 0}});

        ASSERT_EQ(Bx.size(0), 5);
        ASSERT_EQ(Bx.size(1), 7);
        ASSERT_EQ(Bx.size(-1), 7);
        ASSERT_EQ(Bx.size(-2), 5);
        ASSERT_THROW(Bx.size(2), Error);
        ASSERT_THROW(Bx.size(-3), Error);
      }
      {
        // multiple batch dims not at front
        Tensor x = randn({2, 3, 5, 7, 11});
        Tensor Bx = makeBatched(x, {{0, 3}, {1, 1}});

        ASSERT_EQ(Bx.size(0), 2);
        ASSERT_EQ(Bx.size(1), 5);
        ASSERT_EQ(Bx.size(2), 11);
        ASSERT_EQ(Bx.size(-1), 11);
        ASSERT_EQ(Bx.size(-2), 5);
        ASSERT_EQ(Bx.size(-3), 2);
        ASSERT_THROW(Bx.size(3), Error);
        ASSERT_THROW(Bx.size(-4), Error);
      }

    */
}

#[test] fn vmap_test_physical_view_get_shape() {
    todo!();
    /*
    
      {
        VmapPhysicalView physical_view(ones({2, 3, 4, 5, 6}), 1 | 4);
        ASSERT_EQ(physical_view.getPhysicalShape({}), VmapDimVector({2, 3}));
        ASSERT_EQ(physical_view.getPhysicalShape({7}), VmapDimVector({2, 3, 7}));
        ASSERT_EQ(physical_view.getPhysicalShape({7, 11, 13}), VmapDimVector({2, 3, 7, 11, 13}));
        ASSERT_EQ(physical_view.getPhysicalShape({7, 11, 13, 17}), VmapDimVector({2, 3, 7, 11, 13, 17}));
      }
      {
        VmapPhysicalView physical_view(ones({2, 3, 4, 5, 6}), 2);
        ASSERT_EQ(physical_view.getPhysicalShape({}), VmapDimVector({2}));
        ASSERT_EQ(physical_view.getPhysicalShape({7}), VmapDimVector({2, 7}));
      }

    */
}

// Basic test for BatchedTensor::expand
#[test] fn vmap_test_batched_tensor_expand() {
    todo!();
    /*
    
      {
        // Expand size is too small
        auto tensor = randn({2, 3, 5});
        auto batched = makeBatched(tensor, {{/*lvl*/0, /*dim*/0}});
        ASSERT_THROW(batched.expand({5}), Error);
      }
      {
        // Expand size has same dimensionality as the logical dim
        auto tensor = randn({2, 1, 5});
        auto batched = makeBatched(tensor, {{/*lvl*/0, /*dim*/0}});
        auto batched_out = batched.expand({3, 5});
        const auto& out = maybeGetBatchedImpl(batched_out)->value();

        ASSERT_EQ(out.data_ptr(), tensor.data_ptr());
        ASSERT_TRUE(allclose(out, tensor.expand({2, 3, 5})));
      }
      {
        // Expand size has same dimensionality as the logical dim, incorrect expand size
        auto tensor = randn({2, 1, 5});
        auto batched = makeBatched(tensor, {{/*lvl*/0, /*dim*/0}});
        ASSERT_THROW(batched.expand({1, 25}), Error);
      }
      {
        // Expand size has greater dimensionality as the logical dim
        auto tensor = randn({2, 3, 5});
        auto batched = makeBatched(tensor, {{/*lvl*/0, /*dim*/0}});
        auto batched_out = batched.expand({7, 3, 5});
        const auto& out = maybeGetBatchedImpl(batched_out)->value();

        ASSERT_EQ(out.data_ptr(), tensor.data_ptr());
        ASSERT_TRUE(allclose(out, tensor.view({2, 1, 3, 5}).expand({2, 7, 3, 5})));
      }
      {
        // Expand size has greater dimensionality as the logical dim, incorrect expand size
        auto tensor = randn({2, 3, 5});
        auto batched = makeBatched(tensor, {{/*lvl*/0, /*dim*/0}});
        ASSERT_THROW(batched.expand({7, 9, 5}), Error);
      }
      {
        // logical dim is 0, expand size has same dimensionality as logical dim
        auto tensor = randn({2, 3});
        auto batched = makeBatched(tensor, {{0, 0}, {1, 1}});
        auto batched_out = batched.expand({});
        const auto& out = maybeGetBatchedImpl(batched_out)->value();
        ASSERT_EQ(out.data_ptr(), tensor.data_ptr());
        ASSERT_TRUE(allclose(out, tensor));
      }
      {
        // logical dim is 0, expand size has greater dimensionality than logical dim
        auto tensor = randn({2, 3});
        auto batched = makeBatched(tensor, {{0, 0}, {1, 1}});
        auto batched_out = batched.expand({5, 7});
        const auto& out = maybeGetBatchedImpl(batched_out)->value();
        ASSERT_EQ(out.data_ptr(), tensor.data_ptr());
        ASSERT_TRUE(allclose(out, tensor.view({2, 3, 1, 1}).expand({2, 3, 5, 7})));
      }

    */
}

// Basic test for BatchedTensor::unsqueeze
#[test] fn vmap_test_batched_tensor_unsqueeze() {
    todo!();
    /*
    
      {
        // Basic test
        auto tensor = randn({2, 3, 5});  // NOLINT
        auto batched = makeBatched(tensor, {{/*lvl*/0, /*dim*/0}});

        auto batched_out = batched.unsqueeze(0);
        const auto& out = maybeGetBatchedImpl(batched_out)->value();
        ASSERT_EQ(out.data_ptr(), tensor.data_ptr());
        ASSERT_TRUE(allclose(out, tensor.unsqueeze(1)));
      }
      {
        // Test with multiple levels
        auto tensor = randn({2, 3, 5});  // NOLINT
        auto batched = makeBatched(tensor, {{0, 0}, {1, 1}});

        auto batched_out = batched.unsqueeze(0);
        const auto& out = maybeGetBatchedImpl(batched_out)->value();
        ASSERT_EQ(out.data_ptr(), tensor.data_ptr());
        ASSERT_TRUE(allclose(out, tensor.unsqueeze(2)));
      }
      {
        // Negative dim
        auto tensor = randn({2, 3, 5});  // NOLINT
        auto batched = makeBatched(tensor, {{/*lvl*/0, /*dim*/0}});

        auto batched_out = batched.unsqueeze(-1);
        const auto& out = maybeGetBatchedImpl(batched_out)->value();
        ASSERT_EQ(out.data_ptr(), tensor.data_ptr());
        ASSERT_TRUE(allclose(out, tensor.unsqueeze(-1)));
      }

    */
}

// Basic test for BatchedTensor::squeeze(dim)
#[test] fn vmap_test_batched_tensor_squeeze() {
    todo!();
    /*
    
      {
        // Basic test
        auto tensor = randn({2, 1, 5});  // NOLINT
        auto batched = makeBatched(tensor, {{/*lvl*/0, /*dim*/0}});

        auto batched_out = batched.squeeze(0);
        const auto& out = maybeGetBatchedImpl(batched_out)->value();
        ASSERT_EQ(out.data_ptr(), tensor.data_ptr());
        ASSERT_TRUE(allclose(out, tensor.squeeze(1)));
      }
      {
        // Test with multiple levels
        auto tensor = randn({2, 3, 1});  // NOLINT
        auto batched = makeBatched(tensor, {{0, 0}, {1, 1}});

        auto batched_out = batched.squeeze(0);
        const auto& out = maybeGetBatchedImpl(batched_out)->value();
        ASSERT_EQ(out.data_ptr(), tensor.data_ptr());
        ASSERT_TRUE(allclose(out, tensor.squeeze(2)));
      }
      {
        // Negative dim
        auto tensor = randn({2, 3, 1});  // NOLINT
        auto batched = makeBatched(tensor, {{/*lvl*/0, /*dim*/0}});

        auto batched_out = batched.squeeze(-1);
        const auto& out = maybeGetBatchedImpl(batched_out)->value();
        ASSERT_EQ(out.data_ptr(), tensor.data_ptr());
        ASSERT_TRUE(allclose(out, tensor.squeeze(-1)));
      }

    */
}

// Basic test for BatchedTensor::transpose
#[test] fn vmap_test_batched_tensor_transpose() {
    todo!();
    /*
    
      {
        // Basic test
        auto tensor = randn({2, 3, 5});  // NOLINT
        auto batched = makeBatched(tensor, {{/*lvl*/0, /*dim*/0}});

        auto batched_out = batched.transpose(0, 1);
        const auto& out = maybeGetBatchedImpl(batched_out)->value();
        ASSERT_EQ(out.data_ptr(), tensor.data_ptr());
        ASSERT_TRUE(allclose(out, tensor.transpose(1, 2)));
      }
      {
        // Test with multiple levels
        auto tensor = randn({2, 3, 5, 7, 11});  // NOLINT
        auto batched = makeBatched(tensor, {{0, 0}, {1, 1}});

        auto batched_out = batched.transpose(0, 2);
        const auto& out = maybeGetBatchedImpl(batched_out)->value();
        ASSERT_EQ(out.data_ptr(), tensor.data_ptr());
        ASSERT_TRUE(allclose(out, tensor.transpose(2, 4)));
      }
      {
        // Negative dims
        auto tensor = randn({2, 3, 5, 7});  // NOLINT
        auto batched = makeBatched(tensor, {{/*lvl*/0, /*dim*/0}});

        auto batched_out = batched.transpose(-2, -1);
        const auto& out = maybeGetBatchedImpl(batched_out)->value();
        ASSERT_EQ(out.data_ptr(), tensor.data_ptr());
        ASSERT_TRUE(allclose(out, tensor.transpose(-2, -1)));
      }

    */
}

// Basic test for BatchedTensor::permute
#[test] fn vmap_test_batched_tensor_permute() {
    todo!();
    /*
    
      {
        // Basic test
        auto tensor = randn({2, 3, 5});  // NOLINT
        auto batched = makeBatched(tensor, {{/*lvl*/0, /*dim*/0}});

        auto batched_out = batched.permute({1, 0});
        const auto& out = maybeGetBatchedImpl(batched_out)->value();
        ASSERT_EQ(out.data_ptr(), tensor.data_ptr());
        ASSERT_TRUE(allclose(out, tensor.permute({0, 2, 1})));
      }
      {
        // Test with multiple levels
        auto tensor = randn({2, 3, 5, 7, 11});  // NOLINT
        auto batched = makeBatched(tensor, {{0, 0}, {1, 1}});

        auto batched_out = batched.permute({2, 1, 0});
        const auto& out = maybeGetBatchedImpl(batched_out)->value();
        ASSERT_EQ(out.data_ptr(), tensor.data_ptr());
        ASSERT_TRUE(allclose(out, tensor.permute({0, 1, 4, 3, 2})));
      }
      {
        // Negative dims
        auto tensor = randn({2, 3, 5, 7});  // NOLINT
        auto batched = makeBatched(tensor, {{/*lvl*/0, /*dim*/0}});

        auto batched_out = batched.permute({-1, -2, -3});
        const auto& out = maybeGetBatchedImpl(batched_out)->value();
        ASSERT_EQ(out.data_ptr(), tensor.data_ptr());
        ASSERT_TRUE(allclose(out, tensor.permute({0, -1, -2, -3})));
      }

    */
}

pub fn check_multi_batch_vmap_transform(
        inputs:           &[Tensor],
        expected_outputs: &[Tensor])  {
    
    todo!();
        /*
            auto outputs = MultiBatchVmapTransform::logicalToPhysical(inputs);
      ASSERT_EQ(outputs.size(), expected_outputs.size());
      for (i64 idx = 0; idx < outputs.size(); idx++) {
        const auto& output = outputs[idx].tensor();
        ASSERT_EQ(output.data_ptr(), expected_outputs[idx].data_ptr());
        ASSERT_EQ(output.sizes(), expected_outputs[idx].sizes());
        ASSERT_TRUE(allclose(output, expected_outputs[idx]));
      }
        */
}

#[test] fn vmap_test_multi_batch_transform_batched() {
    todo!();
    /*
    
      {
        // Check that batch dims get moved to the front
        i64 B0 = 5, B1 = 7;
        Tensor x = randn({2, B0, 3, B1});
        Tensor y = randn({B1, 2, 3, B0});
        Tensor batched_x = makeBatched(x, {{/*lvl*/0, /*dim*/1}, {/*lvl*/1, /*dim*/3}});
        Tensor batched_y = makeBatched(y, {{/*lvl*/0, /*dim*/3}, {/*lvl*/1, /*dim*/0}});

        checkMultiBatchVmapTransform(
            {batched_x, batched_y},
            {movedim(x, {1, 3}, {0, 1}), movedim(y, {0, 3}, {1, 0})});
      }
      {
        // Check that batch dims become broadcasted and are present in all returns
        i64 B0 = 5, B1 = 7, B2 = 9;
        Tensor x = randn({B0, B2, 2, 3});
        Tensor y = randn({B0, B1, 2, 3});
        Tensor batched_x = makeBatched(x, {{/*lvl*/0, /*dim*/0}, {/*lvl*/2, /*dim*/1}});
        Tensor batched_y = makeBatched(y, {{/*lvl*/0, /*dim*/0}, {/*lvl*/1, /*dim*/1}});

        checkMultiBatchVmapTransform(
            {batched_x, batched_y},
            {x.unsqueeze(1).expand({B0, B1, B2, 2, 3}), y.unsqueeze(2).expand({B0, B1, B2, 2, 3})});
      }
      {
        // Check operation on tensors of different logical dims
        i64 B0 = 5;
        Tensor x = randn({B0, 3});
        Tensor y = randn({B0, 2, 3});
        Tensor batched_x = makeBatched(x, {{/*lvl*/0, /*dim*/0}});
        Tensor batched_y = makeBatched(y, {{/*lvl*/0, /*dim*/0}});

        checkMultiBatchVmapTransform({batched_x, batched_y}, {x, y});
      }
      {
        // More complicated example with two tensors.
        i64 B0 = 5, B1 = 7, B2 = 11, B3 = 13;
        Tensor x = randn({2, B0, 3, B2});
        Tensor y = randn({B3, 3, B1});
        Tensor batched_x = makeBatched(x, {{/*lvl*/0, /*dim*/1}, {/*lvl*/2, /*dim*/3}});
        Tensor batched_y = makeBatched(y, {{/*lvl*/1, /*dim*/2}, {/*lvl*/3, /*dim*/0}});

        checkMultiBatchVmapTransform(
            {batched_x, batched_y},
            {
              x.permute({1, 3, 0, 2}).view({B0, 1, B2, 1, 2, 3}).expand({B0, B1, B2, B3, 2, 3}),
              y.permute({2, 0, 1}).view({1, B1, 1, B3, 3}).expand({B0, B1, B2, B3, 3}),
            });
      }
      {
        // Edge case: BatchedTensor "scalar" handling
        i64 B0 = 5, B2 = 11;
        Tensor x = randn({B0});
        Tensor y = randn({B0, B2});
        Tensor batched_x = makeBatched(x, {{/*lvl*/0, /*dim*/0}});
        Tensor batched_y = makeBatched(y, {{/*lvl*/0, /*dim*/0}, {/*lvl*/1, /*dim*/1}});

        checkMultiBatchVmapTransform({batched_x, batched_y}, {x.view({B0, 1}).expand({B0, B2}), y});
        checkMultiBatchVmapTransform({batched_y, batched_x}, {y, x.view({B0, 1}).expand({B0, B2})});
      }
      {
        // Edge case: Only one tensor is a "batchedtensor scalar"
        i64 B0 = 5, B2 = 11;
        Tensor x = randn({B0});
        Tensor y = randn({B0, B2, 2});
        Tensor batched_x = makeBatched(x, {{/*lvl*/0, /*dim*/0}});
        Tensor batched_y = makeBatched(y, {{/*lvl*/0, /*dim*/0}, {/*lvl*/1, /*dim*/1}});

        checkMultiBatchVmapTransform({batched_x, batched_y}, {x.view({B0, 1}).expand({B0, B2}), y});
        checkMultiBatchVmapTransform({batched_y, batched_x}, {y, x.view({B0, 1}).expand({B0, B2})});
      }

    */
}

#[test] fn vmap_test_multi_batch_transform_batched_unbatched() {
    todo!();
    /*
    
      {
        // Check same example size
        i64 B0 = 5, B1 = 7;
        Tensor x = randn({2, B0, 3, B1});
        Tensor y = randn({2, 3});
        Tensor batched_x = makeBatched(x, {{/*lvl*/0, /*dim*/1}, {/*lvl*/1, /*dim*/3}});

        checkMultiBatchVmapTransform(
            {batched_x, y},
            {movedim(x, {1, 3}, {0, 1}), y.view({1, 1, 2, 3}).expand({B0, B1, 2, 3})});
        checkMultiBatchVmapTransform(
            {y, batched_x},
            {y.view({1, 1, 2, 3}).expand({B0, B1, 2, 3}), movedim(x, {1, 3}, {0, 1})});
      }
      {
        // BatchedTensor has higher example dim than non-batched-tensor
        i64 B0 = 5, B1 = 7;
        Tensor x = randn({B0, B1, 2, 3});
        Tensor y = randn({3});
        Tensor batched_x = makeBatched(x, {{/*lvl*/0, /*dim*/0}, {/*lvl*/1, /*dim*/1}});

        checkMultiBatchVmapTransform(
            {batched_x, y}, {x, y.view({1, 1, 3}).expand({B0, B1, 3})});
        checkMultiBatchVmapTransform(
            {y, batched_x}, {y.view({1, 1, 3}).expand({B0, B1, 3}), x});
      }
      {
        // BatchedTensor has lower example dim than non-batched-tensor
        i64 B0 = 5, B1 = 7;
        Tensor x = randn({B0, B1, 3});
        Tensor y = randn({2, 3});
        Tensor batched_x = makeBatched(x, {{/*lvl*/0, /*dim*/0}, {/*lvl*/1, /*dim*/1}});

        checkMultiBatchVmapTransform(
            {batched_x, y}, {x.view({B0, B1, 3}), y.view({1, 1, 2, 3}).expand({B0, B1, 2, 3})});
        checkMultiBatchVmapTransform(
            {y, batched_x}, {y.view({1, 1, 2, 3}).expand({B0, B1, 2, 3}), x.view({B0, B1, 3})});
      }
      {
        // Scalar handling
        i64 B0 = 5, B1 = 7;
        Tensor x = randn({B0, B1});
        Tensor y = randn({});
        Tensor batched_x = makeBatched(x, {{/*lvl*/0, /*dim*/0}, {/*lvl*/1, /*dim*/1}});

        checkMultiBatchVmapTransform({batched_x, y}, {x, y.view({1, 1}).expand({B0, B1})});
        checkMultiBatchVmapTransform({y, batched_x}, {y.view({1, 1}).expand({B0, B1}), x});
      }

    */
}

#[test] fn vmap_test_multi_batch_transform_max_levels() {
    todo!();
    /*
    
      {
        // inputs have all 64 levels
        auto x = randn(vector<i64>(kVmapNumLevels, 1));
        auto y = randn(vector<i64>(kVmapNumLevels, 1));
        auto batched_x = makeBatched(x, maxBatchDimsAtFront());
        auto batched_y = makeBatched(y, maxBatchDimsAtFront());

        checkMultiBatchVmapTransform({batched_x, batched_y}, {x, y});
      }
      {
        // inputs don't have all 64 levels, but results do.
        i64 split = 19;
        auto x = randn(vector<i64>(split, 1));
        auto y = randn(vector<i64>(kVmapNumLevels - split, 1));

        auto tmp = maxBatchDimsAtFront();
        BatchDims x_bdims(tmp.begin(), tmp.begin() + split);

        // Construct y_bdims.
        i64 dim = 0;
        auto y_bdims_vector = fmap(
            ArrayRef<BatchDim>(tmp.begin() + split, tmp.end()),
            [&](const BatchDim& bdim) -> BatchDim {
              return { bdim.level(), dim++ };
            });
        BatchDims y_bdims(y_bdims_vector.begin(), y_bdims_vector.end());

        auto batched_x = makeBatched(x, x_bdims);
        auto batched_y = makeBatched(y, y_bdims);

        auto expected_size = vector<i64>(kVmapNumLevels, 1);
        checkMultiBatchVmapTransform(
            {batched_x, batched_y},
            {x.view(expected_size), y.view(expected_size)});
      }

    */
}

#[test] fn vmap_test_multi_batch_transform_multiple_tensors() {
    todo!();
    /*
    
      // Test with three (all batched) tensors
      {
        i64 B0 = 5, B1 = 7, B2 = 9;
        Tensor x = randn({2, B0, 3, B1});
        Tensor y = randn({B1, 4});
        Tensor z = randn({2, B2});
        Tensor batched_x = makeBatched(x, {{/*lvl*/0, /*dim*/1}, {/*lvl*/1, /*dim*/3}});
        Tensor batched_y = makeBatched(y, {{/*lvl*/1, /*dim*/0}});
        Tensor batched_z = makeBatched(z, {{/*lvl*/2, /*dim*/1}});

        checkMultiBatchVmapTransform(
            {batched_x, batched_y, batched_z},
            {
              movedim(x, {1, 3}, {0, 1}).view({B0, B1, 1, 2, 3}).expand({B0, B1, B2, 2, 3}),
              y.view({1, B1, 1, 4}).expand({B0, B1, B2, 4}),
              z.t().view({1, 1, B2, 2}).expand({B0, B1, B2, 2}),
            });
      }
      // Test with three tensors, some batched, some unbatched
      {
        i64 B0 = 5, B1 = 7, B2 = 9;
        Tensor x = randn({2, 3});
        Tensor y = randn({4, B0});
        Tensor z = randn({B1, 2, B2});
        Tensor batched_y = makeBatched(y, {{/*lvl*/0, /*dim*/1}});
        Tensor batched_z = makeBatched(z, {{/*lvl*/1, /*dim*/0}, {/*lvl*/2, /*dim*/2}});

        checkMultiBatchVmapTransform(
            {x, batched_y, batched_z},
            {
              x.view({1, 1, 1, 2, 3}).expand({B0, B1, B2, 2, 3}),
              y.t().view({B0, 1, 1, 4}).expand({B0, B1, B2, 4}),
              z.permute({0, 2, 1}).view({1, B1, B2, 2}).expand({B0, B1, B2, 2}),
            });
      }

    */
}
