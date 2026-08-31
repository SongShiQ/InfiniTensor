"""Run a model whose input shape is not known until the moment it is given.

An ONNX dimension is either a number or a name. A named one -- `batch`, say --
carries no number at all, so reading shapes with `dim.dim_value` yields a zero
for it and an empty tensor after that. Such a model needs a shape supplied per
inference instead, which is what `set_input` is for.

Two demo models are built in. The first varies its batch size; the second
varies the height and width of a picture, which the model reads back out of a
pooled tensor to restore the two axes it flattened. The second is the harder
case, and the one where a chain that only ever reads a leading dimension will
not do.

Usage:
    python dynamic_shape_inference.py
    python dynamic_shape_inference.py batch                # just the first
    python dynamic_shape_inference.py image                # just the second
    python dynamic_shape_inference.py model.onnx           # a model of your own
    python dynamic_shape_inference.py model.onnx batch=4 height=32 width=32

For a model of your own, name each dynamic dimension and the sizes to try for
it. A bare number stands for every named dimension at once, which is what a
model with a single dynamic dimension wants. Dimensions the model gives a
number to are kept as they are.
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from pyinfinitensor import backend
from pyinfinitensor.onnx import OnnxStub


def batch_model():
    """A model that reads its own batch size, works on it, and restores it.

    Neither `Reshape` can be worked out when the graph is built: both targets
    come from the `Shape` of the input, which is a different number on every
    inference.
    """
    weight = (np.arange(36, dtype=np.float32).reshape(6, 6) / 18 - 1) / 6
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Shape", ["x"], ["s"]),
                helper.make_node("Gather", ["s", "first"], ["batch"], axis=0),
                helper.make_node("Unsqueeze", ["batch", "zero"], ["batch_1d"]),
                helper.make_node("Concat", ["batch_1d", "six"], ["flat"], axis=0),
                helper.make_node("Reshape", ["x", "flat"], ["rows"]),
                helper.make_node("MatMul", ["rows", "weight"], ["hidden"]),
                helper.make_node("Relu", ["hidden"], ["activated"]),
                helper.make_node(
                    "Concat", ["batch_1d", "two", "three"], ["restored"], axis=0
                ),
                helper.make_node("Reshape", ["activated", "restored"], ["y"]),
            ],
            "dynamic_batch",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 2, 3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 2, 3])],
            [
                numpy_helper.from_array(np.asarray(0, np.int64), "first"),
                numpy_helper.from_array(np.asarray([0], np.int64), "zero"),
                numpy_helper.from_array(np.asarray([6], np.int64), "six"),
                numpy_helper.from_array(np.asarray([2], np.int64), "two"),
                numpy_helper.from_array(np.asarray([3], np.int64), "three"),
                numpy_helper.from_array(weight, "weight"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )


def image_model():
    """A model over pictures of no fixed size.

    A convolution and a pooling both change the spatial size, so by the time
    the shape subgraph reads it, the height and the width are numbers no part
    of the graph was given. Both are taken out separately, the two of them are
    flattened into one axis so a softmax can run along a length only this chain
    knows, and both are then put back.
    """
    rng = np.random.default_rng(20260831)
    first = (rng.standard_normal((8, 3, 3, 3)) / 5).astype(np.float32)
    second = (rng.standard_normal((4, 8, 3, 3)) / 5).astype(np.float32)
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Conv", ["x", "w1"], ["c1"], pads=[1, 1, 1, 1]),
                helper.make_node("Relu", ["c1"], ["r1"]),
                helper.make_node(
                    "MaxPool", ["r1"], ["p"], kernel_shape=[2, 2], strides=[2, 2]
                ),
                helper.make_node("Conv", ["p", "w2"], ["c2"], pads=[1, 1, 1, 1]),
                helper.make_node("Relu", ["c2"], ["r2"]),
                helper.make_node("Shape", ["r2"], ["s"]),
                helper.make_node("Gather", ["s", "i0"], ["n"], axis=0),
                helper.make_node("Gather", ["s", "i2"], ["h"], axis=0),
                helper.make_node("Gather", ["s", "i3"], ["w"], axis=0),
                helper.make_node("Unsqueeze", ["n", "zero"], ["n1"]),
                helper.make_node("Unsqueeze", ["h", "zero"], ["h1"]),
                helper.make_node("Unsqueeze", ["w", "zero"], ["w1d"]),
                helper.make_node("Concat", ["n1", "four", "minus1"], ["flat"], axis=0),
                helper.make_node("Reshape", ["r2", "flat"], ["rows"]),
                helper.make_node("Softmax", ["rows"], ["soft"], axis=-1),
                helper.make_node(
                    "Concat", ["n1", "four", "h1", "w1d"], ["back"], axis=0
                ),
                helper.make_node("Reshape", ["soft", "back"], ["y"]),
            ],
            "dynamic_image",
            [
                helper.make_tensor_value_info(
                    "x", TensorProto.FLOAT, ["batch", 3, "height", "width"]
                )
            ],
            [
                helper.make_tensor_value_info(
                    "y", TensorProto.FLOAT, ["batch", 4, "out_h", "out_w"]
                )
            ],
            [
                numpy_helper.from_array(first, "w1"),
                numpy_helper.from_array(second, "w2"),
                numpy_helper.from_array(np.asarray(0, np.int64), "i0"),
                numpy_helper.from_array(np.asarray(2, np.int64), "i2"),
                numpy_helper.from_array(np.asarray(3, np.int64), "i3"),
                numpy_helper.from_array(np.asarray([0], np.int64), "zero"),
                numpy_helper.from_array(np.asarray([4], np.int64), "four"),
                numpy_helper.from_array(np.asarray([-1], np.int64), "minus1"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )


def shape_of(value_info, sizes, fallback):
    """The shape to ask for, with every dimension lacking a number filled in.

    A dimension holds a number or it does not; asking which field is set is the
    whole test. Reading `dim_value` off one that carries a name instead returns
    a zero, which is a shape a tensor can genuinely have and so passes quietly.

    A named dimension takes its size from `sizes` under that name, and
    `fallback` otherwise. Giving every named dimension the same number is only
    right when there is one of them: a picture whose height and width both came
    from a batch size would be square by accident.
    """
    shape = []
    for d in value_info.type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            shape.append(d.dim_value)
        else:
            shape.append(sizes.get(d.dim_param, fallback))
    return shape


def reference(model, feeds):
    """What onnxruntime makes of the same inputs, if it is installed."""
    try:
        from onnxruntime import InferenceSession
    except ImportError:
        return None
    session = InferenceSession(model.SerializeToString())
    return session.run(None, feeds)[0]


# What to try for each built-in model. The sizes are picked for what they
# exercise: a repeat takes the path where nothing has to be laid out again, and
# a shape smaller than an earlier one is where a buffer kept from the larger one
# would still read as big enough.
DEMOS = {
    "batch": (
        batch_model,
        [{"batch": n} for n in (1, 4, 4, 16, 2)],
    ),
    "image": (
        image_model,
        [
            {"batch": 1, "height": 8, "width": 8},
            {"batch": 1, "height": 8, "width": 8},
            {"batch": 1, "height": 8, "width": 12},
            {"batch": 2, "height": 12, "width": 8},
            {"batch": 1, "height": 4, "width": 4},
        ],
    ),
}


def parse(argv):
    """Work out which models to run and at which sizes.

    Returns a list of (label, model, schedule) where a schedule is a list of
    dimension-name to size mappings, one per inference.
    """
    args = argv[1:]
    if args and args[0] in DEMOS:
        build, schedule = DEMOS[args[0]]
        return [(args[0], build(), schedule)]
    if not args:
        return [(name, build(), schedule) for name, (build, schedule) in DEMOS.items()]

    model = onnx.load(args[0])
    named, bare = {}, []
    for arg in args[1:]:
        if "=" in arg:
            name, _, value = arg.partition("=")
            named.setdefault(name, []).append(int(value))
        else:
            bare.append(int(arg))
    if named:
        # Each named dimension is stepped through its own sizes together with
        # the others, so the shortest list decides how many inferences run.
        rounds = min(len(v) for v in named.values())
        schedule = [{k: v[i] for k, v in named.items()} for i in range(rounds)]
    else:
        schedule = [{"": n} for n in bare or [1, 4, 16]]
    return [(args[0], model, schedule)]


def run(label, model, schedule):
    inputs = list(model.graph.input)
    stub = OnnxStub(model, backend.cpu_runtime())
    rng = np.random.default_rng(0)

    for sizes in schedule:
        fallback = next(iter(sizes.values()))
        shapes = [shape_of(i, sizes, fallback) for i in inputs]

        # Hand the graph the shapes it could not know, which re-infers every
        # shape downstream of them -- including whatever a Reshape reads from a
        # Shape subgraph.
        stub.set_input(shapes)

        feeds = {}
        for value_info, shape in zip(inputs, shapes):
            data = rng.standard_normal(shape).astype(np.float32)
            stub.tensors[value_info.name].copyin_numpy(data)
            feeds[value_info.name] = data

        stub.run()

        name, output = next(iter(stub.outputs.items()))
        got = np.asarray(output.copyout_float(), dtype=np.float32)
        # An unnamed size stands for every dynamic dimension at once, so there
        # is no name worth printing beside it.
        asked = ", ".join(f"{k}={v}" if k else str(v) for k, v in sizes.items())
        line = f"{label:>6} [{asked}]: {name} has shape {output.shape()}"

        want = reference(model, feeds)
        if want is not None and want.size == got.size:
            error = float(np.abs(got.reshape(want.shape) - want).max())
            line += f", differs from onnxruntime by at most {error:.2e}"
        print(line)


def main(argv):
    for label, model, schedule in parse(argv):
        run(label, model, schedule)


if __name__ == "__main__":
    main(sys.argv)
