#include "operators/matmul.h"
#include "core/kernel.h"
#include <numeric>

namespace infini {

class NaiveMatmul : public CpuKernelWithoutConfig {
    /// How many matrices an operand carries, being the product of everything
    /// before its last two dimensions.
    static int batchOf(const Tensor &tensor) {
        const auto &dims = tensor->getDims();
        if (dims.size() <= 2) {
            return 1;
        }
        return std::accumulate(dims.begin(), dims.end() - 2, 1,
                               std::multiplies<int>());
    }

    template <typename T>
    void doCompute(const Operator &_op, const RuntimeObj *context) const {
        auto op = as<MatmulObj>(_op);
        IT_ASSERT(op->getInputs().size() == 2, "Bias is not supported yet.");
        T *A = op->getInputs(0)->getRawDataPtr<T *>();
        T *B = op->getInputs(1)->getRawDataPtr<T *>();
        T *C = op->getOutput()->getRawDataPtr<T *>();
        IT_ASSERT(op->getTransA() == false && op->getTransB() == false);
        IT_ASSERT(op->getAct() == ActType::None);
        const int M = op->getM(), N = op->getN(), K = op->getK();

        // Everything before the last two dimensions is a stack of matrices to
        // work through. The operator has already broadcast those dimensions
        // together and reports the total, so the loop follows its count rather
        // than working the shapes out again.
        const int batch = op->getB();
        const int batchA = batchOf(op->getInputs(0));
        const int batchB = batchOf(op->getInputs(1));
        // An operand either carries a matrix for every one of them or a single
        // matrix meant for all of them. Anything else would be a broadcast this
        // does not describe, and reading it as either would be wrong.
        IT_ASSERT(batchA == batch || batchA == 1);
        IT_ASSERT(batchB == batch || batchB == 1);
        // A single matrix is reread for each one, which is what a zero stride
        // says.
        const int strideA = batchA == 1 ? 0 : M * K;
        const int strideB = batchB == 1 ? 0 : K * N;

        for (int p = 0; p < batch; p++) {
            const T *a = A + static_cast<ptrdiff_t>(p) * strideA;
            const T *b = B + static_cast<ptrdiff_t>(p) * strideB;
            T *c = C + static_cast<ptrdiff_t>(p) * M * N;
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    c[i * N + j] = 0;
                    for (int k = 0; k < K; k++) {
                        c[i * N + j] += a[i * K + k] * b[k * N + j];
                    }
                }
            }
        }
    }

    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
#define CASE(N)                                                                \
    case N:                                                                    \
        doCompute<DT<N>::t>(_op, context)

        int dataTypeIdx = _op->getDType().getIndex();
        switch (dataTypeIdx) {
            CASE(1); // DataType::Float32
            break;
            CASE(12); // DataType::UInt32
            break;
        default:
            IT_TODO_HALT();
        }
    }
};

REGISTER_KERNEL(Device::CPU, OpType::MatMul, NaiveMatmul, "MatmulNaive_CPU");

} // namespace infini
