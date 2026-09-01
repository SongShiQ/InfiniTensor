#include "operators/slice.h"

namespace infini {
SliceObj::SliceObj(GraphObj *graph, Tensor input, Tensor output,
                   const vector<int> &starts, const vector<int> &ends,
                   const optional<vector<int>> &_axes,
                   const optional<vector<int>> &_steps)
    : OperatorObj(OpType::Slice, {input}, {output}) {
    auto shape = input->getDims(); // shape of input
    map<size_t, size_t> axes;
    vector<int> steps;
    {
        auto size = starts.size();      // size of starts
        IT_ASSERT(size == ends.size()); // size of ends

        if (_axes) {
            IT_ASSERT(size == _axes->size());
            // onnx doc: "Behavior is undefined if an axis is repeated."
            IT_ASSERT(size == std::set(_axes->begin(), _axes->end()).size());

            for (size_t i = 0; i < size; ++i) {
                auto index = _axes->at(i);
                if (index < 0)
                    index += shape.size();
                axes[index] = i;
            }
        } else
            for (size_t i = 0; i < size; ++i)
                axes[i] = i;

        if (_steps) {
            IT_ASSERT(size == _steps->size());
            // onnx doc: "‘steps’ cannot be 0."
            IT_ASSERT(std::find(_steps->begin(), _steps->end(), 0) ==
                      _steps->end());
            steps = *_steps;
        } else {
            steps.reserve(size);
            for (size_t i = 0; i < size; ++i)
                steps.push_back(1);
        }
    }

    auto size = shape.size();
    this->axes.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        auto len = shape[i];
        if (auto _i = axes.find(i); _i != axes.end()) {
            auto __i = _i->second;
            auto start = starts[__i];
            auto end = ends[__i];
            if (start > len)
                start = len;
            if (end > len)
                end = len;
            this->axes.push_back({start >= 0 ? start : start + len,
                                  end >= 0 ? end : end + len, steps[__i]});
        } else {
            this->axes.push_back({0, len, 1});
        }
    }
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> SliceObj::inferShape(const TensorVec &inputs) {
    Shape ans;
    ans.reserve(axes.size());
    for (const auto &range : axes) {
        auto step = std::abs(range.step);
        ans.push_back((range.end - range.start + step - 1) / step);
    }
    return {{ans}};
}

void SliceObj::inferShapeValue() {
    if (!beginShapeValueUpdate()) {
        return;
    }
    // A shape is a list, and taking part of a list is the one case worth
    // handling: a model that reshapes to "everything but the last dimension,
    // then whatever is left" writes exactly this, and a simplifier will produce
    // it even where the model did not. Slicing anything of higher rank is a
    // slice of data rather than of dimensions, and nothing here can say what
    // the result would be.
    if (inputs[0]->getRank() != 1 || axes.size() != 1) {
        return;
    }
    const auto &value = *inputs[0]->getShapeValue();
    const auto &range = axes[0];
    // Ends past the end were already clamped to the length when this operator
    // was built, and negative starts were resolved against it. What is left is
    // to walk the range, which a negative step walks backwards.
    vector<int64_t> picked;
    vector<bool> fixed;
    for (int i = range.start; range.step > 0 ? i < range.end : i > range.end;
         i += range.step) {
        if (i < 0 || i >= static_cast<int>(value.size())) {
            return;
        }
        picked.push_back(value[i]);
        fixed.push_back(inputs[0]->isShapeValueFixed(i));
    }
    // Whatever the range worked out to has to be what the output was shaped
    // for; anything else means the two disagree and the value is not usable.
    if (picked.size() != outputs[0]->size()) {
        return;
    }
    outputs[0]->setShapeValue(std::move(picked), std::move(fixed));
}

std::string SliceObj::toString() const {
    std::ostringstream os;
    os << "Slice[" << getGuid() << "][";
    for (const auto &range : axes) {
        os << range.start << ':' << range.step << ':' << range.end << ", ";
    }
    os << "]("
       << "input=" << inputs[0]->getGuid() << ", "
       << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> SliceObj::getWorkloadVector() const {
    auto ans = getOpAttrVector();
    {
        auto i = inputs[0]->getDims();
        ans.insert(ans.end(), i.begin(), i.end());
    }
    if (!outputs.empty()) {
        auto o = outputs[0]->getDims();
        ans.insert(ans.end(), o.begin(), o.end());
    }
    return ans;
}

vector<int> SliceObj::getOpAttrVector() const {
    vector<int> ans{type.underlying()};
    for (const auto &range : axes) {
        ans.push_back(range.start);
        ans.push_back(range.end);
        ans.push_back(range.step);
    }
    return ans;
}

} // namespace infini
