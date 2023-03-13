crate::ix!();

#[test] fn boolean_mask_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "BooleanMask",
        ["data", "mask"],
        ["masked_data", "masked_indices"]
    )

    workspace.FeedBlob("data", np.array([1,2,3,4,5,6]))
    workspace.FeedBlob("mask", np.array([True,False,False,True,True,False]))
    print("data:", workspace.FetchBlob("data"))
    print("mask:", workspace.FetchBlob("mask"))
    workspace.RunOperatorOnce(op)
    print("masked_data:", workspace.FetchBlob("masked_data"))
    print("masked_indices:", workspace.FetchBlob("masked_indices"))

    result:
    data: [1 2 3 4 5 6]
    mask: [ True False False  True  True False]
    masked_data: [1 4 5]
    masked_indices: [0 3 4]
    */
}

#[test] fn boolean_mask_lengths_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "BooleanMaskLengths",
        ["lengths", "mask"],
        ["masked_lengths"]
    )

    workspace.FeedBlob("lengths", np.array([1,3,2], dtype=np.int32))
    workspace.FeedBlob("mask", np.array([False,True,True,False,True,True]))
    print("lengths:", workspace.FetchBlob("lengths"))
    print("mask:", workspace.FetchBlob("mask"))
    workspace.RunOperatorOnce(op)
    print("masked_lengths:", workspace.FetchBlob("masked_lengths"))



    lengths: [1 3 2]
    mask: [False  True  True False  True  True]
    masked_lengths: [0 2 2]
    */
}

///A comprehensive example:
#[test] fn boolean_unmask_op_example1() {

    todo!();
    /*
    mask1   = True, False, True, False, False
    values1 = 1.0, 3.0
    mask2   = False, True, False, False, False
    values2 = 2.0
    mask3   = False, False, False, True, True
    values3 = 4.0, 5.0

    Reconstruct by:

    output = net.BooleanUnmask([mask1, values1, mask2, values2, mask3, values3], ["output"])
    output = 1.0, 2.0, 3.0, 4.0, 5.0
    */
}

/**
  | @note
  | 
  | for all mask positions, there must be
  | at least one True. This is not allowed:
  |
  */
#[test] fn boolean_unmask_op_example2() {

    todo!();
    /*
    mask1   = True, False
    values1 = 1.0
    mask2   = False, False
    values2 =

    output = net.BooleanUnmask([mask1, values1, mask2, values2], ["output"])
    */
}

/**
  |If there are multiple True values for a field,
  |we accept the first value, and no longer expect
  |a value for that location:
  |
  |*** Note that we alternate `data` and `mask`
  |inputs
  */
#[test] fn boolean_unmask_op_example3() {

    todo!();

    /*
    mask1   = True, False
    values1 = 1.0
    mask2   = True, True
    values2 = 2.0

    output = net.BooleanUnmask([mask1, values1, mask2, values2], ["output"])
    output = 1.0, 2.0
    */
}

#[test] fn boolean_unmask_op_example4() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "BooleanUnmask",
        ["mask1", "data1", "mask2", "data2"],
        ["unmasked_data"]
    )

    workspace.FeedBlob("mask1", np.array([True,False,False,True,True,False]))
    workspace.FeedBlob("data1", np.array([1,4,5]))
    workspace.FeedBlob("mask2", np.array([False,True,True,False,False,True]))
    workspace.FeedBlob("data2", np.array([2,3,6]))

    print("data1:", workspace.FetchBlob("data1"))
    print("mask1:", workspace.FetchBlob("mask1"))
    print("data2:", workspace.FetchBlob("data2"))
    print("mask2:", workspace.FetchBlob("mask2"))
    workspace.RunOperatorOnce(op)
    print("unmasked_data:", workspace.FetchBlob("unmasked_data"))

    data1: [1 4 5]
    mask1: [ True False False  True  True False]
    data2: [2 3 6]
    mask2: [False  True  True False False  True]
    unmasked_data: [1 2 3 4 5 6]

    */
}

/**
  | Test case for BooleanUnmask operator
  |  mask1:   [ false ]
  |  values1: [ ]
  |  mask2:   [ true ]
  |  values2: [ 1.0 ]
  |
  |  Expected Output: [ 1.0 ]
  */
#[test] fn boolean_unmask_test() {
    todo!();
    /*
  Workspace ws;
  OperatorDef def;

  def.set_name("test");
  def.set_type("BooleanUnmask");

  def.add_input("mask1");
  def.add_input("values1");
  def.add_input("mask2");
  def.add_input("values2");

  def.add_output("unmasked_data");

  AddScalarInput(false, "mask1", &ws);
  AddScalarInput(float(), "values1", &ws, true);
  AddScalarInput(true, "mask2", &ws);
  AddScalarInput(1.0f, "values2", &ws);

  unique_ptr<OperatorStorage> op(CreateOperator(def, &ws));
  EXPECT_NE(nullptr, op.get());

  EXPECT_TRUE(op->Run());

  Blob* unmasked_data_blob = ws.GetBlob("unmasked_data");
  EXPECT_NE(nullptr, unmasked_data_blob);

  auto& unmasked_data = unmasked_data_blob->Get<TensorCPU>();
  EXPECT_EQ(unmasked_data.numel(), 1);

  CHECK_EQ(unmasked_data.data<float>()[0], 1.0f);
  */
}
