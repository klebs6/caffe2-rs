crate::ix!();

const IS_MEMBER_OF_OP_VALUE_TAG: &'static str = "value";

#[test] fn is_member_of_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "IsMemberOf",
        ["X"],
        ["Y"],
        value=[0,2,4,6,8],
    )

    // Use a not-empty tensor
    workspace.FeedBlob("X", np.array([0,1,2,3,4,5,6,7,8]).astype(np.int32))
    print("X:\n", workspace.FetchBlob("X"))

    workspace.RunOperatorOnce(op)
    print("Y: \n", workspace.FetchBlob("Y"))

    **Result**

    // value=[0,2,4,6,8]

    X:
     [0 1 2 3 4 5 6 7 8]
    Y:
     [ True False  True False  True False  True False  True]
    */
}
