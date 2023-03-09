crate:ix!();

#[test] fn lengths_max_extra_op_example() {

    todo!();

    /*

    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "LengthsMax",
        ["DATA", "LENGTHS"],
        ["OUTPUT"],
    )

    workspace.FeedBlob("DATA", np.array([2,4,3,1,2,10]).astype(np.float32))
    print("DATA:\n", workspace.FetchBlob("DATA"))

    workspace.FeedBlob("LENGTHS", np.array([2,3,1]).astype(np.int32))
    print("LENGTHS:\n", workspace.FetchBlob("LENGTHS"))

    workspace.RunOperatorOnce(op)
    print("OUTPUT: \n", workspace.FetchBlob("OUTPUT"))

    DATA:
     [ 2.  4.  3.  1.  2. 10.]
    LENGTHS:
     [2 3 1]
    OUTPUT:
     [ 4.  3. 10.]

    */
}

#[test] fn lengths_mean_extra_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "LengthsMean",
        ["DATA", "LENGTHS"],
        ["OUTPUT"],
    )

    workspace.FeedBlob("DATA", np.array([2,4,3,1,2,10]).astype(np.float32))
    print("DATA:\n", workspace.FetchBlob("DATA"))

    workspace.FeedBlob("LENGTHS", np.array([2,3,1]).astype(np.int32))
    print("LENGTHS:\n", workspace.FetchBlob("LENGTHS"))

    workspace.RunOperatorOnce(op)
    print("OUTPUT: \n", workspace.FetchBlob("OUTPUT"))

    DATA:
     [ 2.  4.  3.  1.  2. 10.]
    LENGTHS:
     [2 3 1]
    OUTPUT:
     [ 3.  2. 10.]
    */
}

#[test] fn lengths_sum_extra_op() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "LengthsSum",
        ["DATA", "LENGTHS"],
        ["OUTPUT"],
    )

    workspace.FeedBlob("DATA", np.array([2,4,3,1,2,10]).astype(np.float32))
    print("DATA:\n", workspace.FetchBlob("DATA"))

    workspace.FeedBlob("LENGTHS", np.array([2,3,1]).astype(np.int32))
    print("LENGTHS:\n", workspace.FetchBlob("LENGTHS"))

    workspace.RunOperatorOnce(op)
    print("OUTPUT: \n", workspace.FetchBlob("OUTPUT"))

    DATA:
     [ 2.  4.  3.  1.  2. 10.]
    LENGTHS:
     [2 3 1]
    OUTPUT:
     [ 6.  6. 10.]
    */
}

/*
 | using LengthsSumCPUOp = AbstractLengthsDef<
 |     float,
 |     int,
 |     CPUContext,
 |     SumReducerDef,
 |     true>::ForwardOp;
 |
 | using LengthsMeanCPUOp = AbstractLengthsDef<
 |     float,
 |     int,
 |     CPUContext,
 |     MeanReducerDef,
 |     true>::ForwardOp;
 */
#[test] fn lengths_weighted_sum_extra_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "LengthsWeightedSum",
        ["DATA", "SCALARS","LENGTHS"],
        ["OUTPUT"],
    )

    workspace.FeedBlob("DATA", np.array([2,4,3,1,2,10]).astype(np.float32))
    print("DATA:\n", workspace.FetchBlob("DATA"))

    workspace.FeedBlob("SCALARS", np.array([8, 2, 1, 4, 1, 0.6]).astype(np.float32))
    print("SCALARS:\n", workspace.FetchBlob("SCALARS"))

    workspace.FeedBlob("LENGTHS", np.array([2,3,1]).astype(np.int32))
    print("LENGTHS:\n", workspace.FetchBlob("LENGTHS"))

    workspace.RunOperatorOnce(op)
    print("OUTPUT: \n", workspace.FetchBlob("OUTPUT"))


    DATA:
     [ 2.  4.  3.  1.  2. 10.]
    SCALARS:
     [8.  2.  1.  4.  1.  0.6]
    LENGTHS:
     [2 3 1]
    OUTPUT:
     [24.  9.  6.]
    */
}
