crate::ix!();

#[test] fn pow_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Pow",
        ["X", "exponent"],
        ["Y"],
        broadcast=1
    )

    workspace.FeedBlob("X", np.array([1,2,3,4,5,6]).astype(np.float32))
    print("X: ", workspace.FetchBlob("X"))

    workspace.FeedBlob("exponent", np.array([2]).astype(np.float32))
    print("exponent: ", workspace.FetchBlob("exponent"))

    workspace.RunOperatorOnce(op)
    print("Y: ", workspace.FetchBlob("Y"))

    X:  [1. 2. 3. 4. 5. 6.]
    exponent:  [2.]
    Y:  [ 1.  4.  9. 16. 25. 36.]

    */
}
