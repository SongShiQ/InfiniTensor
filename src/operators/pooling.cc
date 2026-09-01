#include "operators/pooling.h"
#include <algorithm>

namespace infini {

void PoolingObj::takeInputDims(const Tensor &input) {
    n = input->getDims().at(0);
    c = input->getDims().at(1);
    h = input->getRank() == 3 ? 1 : input->getDims().at(2);
    w = input->getRank() == 3 ? input->getDims().at(2) : input->getDims().at(3);
}

PoolingObj::PoolingObj(GraphObj *graph, OpType optype, Tensor input,
                       Tensor output, int kh, int kw, int dh, int dw, int ph,
                       int pw, int sh, int sw, int ceilMode)
    : OperatorObj(optype, {input}, {output}), kh(kh), kw(kw), dh(dh), dw(dw),
      ph(ph), pw(pw), sh(sh), sw(sw), ceilMode(ceilMode) {
    takeInputDims(input);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> PoolingObj::inferShape(const TensorVec &inputs) {
    const auto &input = inputs[0];
    // The spatial size is read afresh every time rather than kept from
    // construction: a dynamic height or width is only a placeholder until a
    // real shape arrives, and the kernel strides over the input by these same
    // numbers, so a stale one is read as well as reported.
    takeInputDims(input);
    int oh, ow;
    if (ceilMode) {
        oh = ceil(((float)(h + 2 * ph - dh * (kh - 1) - 1)) / sh + 1);
        ow = ceil(((float)(w + 2 * pw - dw * (kw - 1) - 1)) / sw + 1);
    } else {
        oh = floor(((float)(h + 2 * ph - dh * (kh - 1) - 1)) / sh + 1);
        ow = floor(((float)(w + 2 * pw - dw * (kw - 1) - 1)) / sw + 1);
    }
    // A window wider than what it is given still covers a position, and covers
    // it once, so there is one row and one column of results however small the
    // input is. Working the count out arithmetically gives zero or less there,
    // and a spatial size of zero is not a picture the rest of a graph can read:
    // the shape chain after this carries the zero into a reshape target, where
    // ONNX spells a dimension of zero the same way it spells keeping the
    // dimension the input already has. onnxruntime answers one here too.
    oh = std::max(oh, 1);
    ow = std::max(ow, 1);

    auto ret = input->getDims();
    if (input->getRank() == 4) {
        ret[input->getRank() - 2] = oh;
    }
    ret[input->getRank() - 1] = ow;
    return {{ret}};
}

vector<DimSource> PoolingObj::dimSources(size_t output, size_t dim) const {
    IT_ASSERT(output == 0);
    IT_ASSERT(dim < outputs[0]->getRank());
    // The output keeps the rank it was given and each dimension is worked out
    // from the one in its place: batch and channels are passed through, and a
    // spatial one is strided over by a window the attributes fix.
    return {DimSource{0, dim}};
}

std::string PoolingObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << "k=[" << kh << "," << kw << "],";
    os << "p=[" << ph << "," << pw << "],";
    os << "s=[" << sh << "," << sw << "],";
    os << "d=[" << dh << "," << dw << "],";
    os << "ceil mode=" << ceilMode << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> PoolingObj::getWorkloadVector() const {
    return {type.underlying(), n, c, h, w, kh, kw, ph, pw, sh, sw, dh, dw,
            ceilMode};
}

vector<int> PoolingObj::getOpAttrVector() const {
    return {type.underlying(), kh, kw, ph, pw, sh, sw, dh, dw, ceilMode};
}

}; // namespace infini
