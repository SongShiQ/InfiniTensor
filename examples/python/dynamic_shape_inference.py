"""Run a model whose input shape is not known until the moment it is given.

An ONNX dimension is either a number or a name. A named one -- `batch`, say --
carries no number at all, so reading shapes with `dim.dim_value` yields a zero
for it and an empty tensor after that. Such a model needs a shape supplied per
inference instead, which is what `set_input` is for.

Usage:
    python dynamic_shape_inference.py                     # built-in demo model
    python dynamic_shape_inference.py model.onnx          # a model of your own
    python dynamic_shape_inference.py model.onnx 1 4 16   # over these batches

Any dimension the model leaves named is filled with the batch size; dimensions
it gives a number to are kept as they are.
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from pyinfinitensor import backend
from pyinfinitensor.onnx import OnnxStub


def demo_model():
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


def shape_of(value_info, batch):
    """The shape to ask for, with every dimension lacking a number filled in.

    A dimension holds a number or it does not; asking which field is set is the
    whole test. Reading `dim_value` off one that carries a name instead returns
    a zero, which is a shape a tensor can genuinely have and so passes quietly.
    """
    return [
        d.dim_value if d.HasField("dim_value") else batch
        for d in value_info.type.tensor_type.shape.dim
    ]


def reference(model, feeds):
    """What onnxruntime makes of the same inputs, if it is installed."""
    try:
        from onnxruntime import InferenceSession
    except ImportError:
        return None
    session = InferenceSession(model.SerializeToString())
    return session.run(None, feeds)[0]


def main(argv):
    if len(argv) > 1:
        model = onnx.load(argv[1])
        batches = [int(a) for a in argv[2:]] or [1, 4, 16]
    else:
        model = demo_model()
        batches = [int(a) for a in argv[1:]] or [1, 4, 16]

    inputs = list(model.graph.input)
    stub = OnnxStub(model, backend.cpu_runtime())
    rng = np.random.default_rng(0)

    for batch in batches:
        shapes = [shape_of(i, batch) for i in inputs]

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
        line = f"batch {batch:>5}: {name} has shape {output.shape()}"

        want = reference(model, feeds)
        if want is not None and want.size == got.size:
            error = float(np.abs(got.reshape(want.shape) - want).max())
            line += f", differs from onnxruntime by at most {error:.2e}"
        print(line)


if __name__ == "__main__":
    main(sys.argv)
