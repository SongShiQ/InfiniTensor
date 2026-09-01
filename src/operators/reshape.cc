#include "operators/reshape.h"
#include "utils/operator_utils.h"
#include <numeric>

namespace infini {
namespace {
/// Works out the real output shape from a target holding the two placeholders
/// ONNX allows: 0 keeps the input dimension in that position, and -1 stands for
/// whatever is left over. Both ways of giving a target share this, so they
/// cannot drift apart.
Shape resolveTargetShape(const Shape &dims, const Shape &inputShape, int size) {
    int count = 0;
    for (auto x : dims) {
        if (x == -1) {
            count++;
        }
        IT_ASSERT(x == -1 || x >= 0);
    }
    IT_ASSERT(count == 0 || count == 1);
    int index = -1;
    Shape outputShape = dims;
    for (int i = 0; i < (int)dims.size(); ++i) {
        if (dims[i] == 0) {
            outputShape[i] = inputShape[i];
        }
        if (dims[i] == -1) {
            index = i;
        }
    }
    if (index != -1) {
        outputShape[index] =
            size / (-std::accumulate(outputShape.begin(), outputShape.end(), 1,
                                     [](auto acc, auto x) { return acc * x; }));
    }
    int outputSize = std::accumulate(outputShape.begin(), outputShape.end(), 1,
                                     [](auto acc, auto x) { return acc * x; });
    IT_ASSERT(outputSize == size);
    return outputShape;
}
} // namespace

ReshapeObj::ReshapeObj(GraphObj *graph, Tensor input, Tensor output, Shape dims)
    : OperatorObj(OpType::Reshape, {input}, {output}), dims(std::move(dims)) {
    IT_ASSERT(checkValid(graph));
}

ReshapeObj::ReshapeObj(GraphObj *graph, Tensor input, Tensor shape,
                       Tensor output)
    : OperatorObj(OpType::Reshape, {input, shape}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ReshapeObj::inferShape(const TensorVec &inputs) {
    if (inputs.size() == 2) {
        IT_ASSERT(inputs[1]->getShapeValue().has_value(),
                  "the target shape of this Reshape is not known while shapes "
                  "are inferred, so it holds data rather than dimensions");
        dims = inputs[1]->getShapeValueAsShape();
    }
    outputShape = resolveTargetShape(dims, inputs[0]->getDims(),
                                     static_cast<int>(inputs[0]->size()));
    return {{outputShape}};
}

std::string ReshapeObj::toString() const {
    std::ostringstream os;
    os << "Reshape[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "outputShape=" << vecToString(outputShape) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ReshapeObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.insert(ret.end(), outputShape.begin(), outputShape.end());
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}
vector<int> ReshapeObj::getOpAttrVector() const {
    vector<int> ret = outputShape;
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

FlattenObj::FlattenObj(GraphObj *graph, Tensor input, Tensor output, int _axis)
    : OperatorObj(OpType::Flatten, {input}, {output}) {
    int rank = input->getRank();
    axis = get_real_axis(_axis, rank);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> FlattenObj::inferShape(const TensorVec &inputs) {
    int sizeB = 1, sizeE = 1;
    auto dims = getInputs(0)->getDims();
    int rank = getInputs(0)->getRank();
    for (int i = 0; i < rank; ++i) {
        ((i < axis) ? sizeB : sizeE) *= dims.at(i);
    }
    return {{{sizeB, sizeE}}};
}

std::string FlattenObj::toString() const {
    std::ostringstream os;
    os << "Flatten[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ",";
    os << "axis=" << axis << ")";
    return os.str();
}

vector<int> FlattenObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), axis);
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> FlattenObj::getOpAttrVector() const {
    return {type.underlying(), axis};
}

IdentityObj::IdentityObj(GraphObj *graph, Tensor input, Tensor output)
    : OperatorObj(OpType::Identity, {input}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> IdentityObj::inferShape(const TensorVec &inputs) {
    return {{getInputs(0)->getDims()}};
}

void IdentityObj::inferShapeValue() {
    if (!beginShapeValueUpdate()) {
        return;
    }
    // The output is the input: the same elements, each as settled as it was.
    const auto &value = *inputs[0]->getShapeValue();
    vector<bool> fixed;
    fixed.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        fixed.push_back(inputs[0]->isShapeValueFixed(i));
    }
    outputs[0]->setShapeValue(value, std::move(fixed));
}

std::string IdentityObj::toString() const {
    std::ostringstream os;
    os << "Identity[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> IdentityObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}
vector<int> IdentityObj::getOpAttrVector() const { return {type.underlying()}; }
} // namespace infini
