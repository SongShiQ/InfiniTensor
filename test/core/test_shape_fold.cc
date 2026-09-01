#include "core/graph.h"
#include "core/runtime.h"
#include "operators/concat.h"
#include "operators/element_wise.h"
#include "operators/gather.h"
#include "operators/reshape.h"
#include "operators/unary.h"
#include "operators/unsqueeze.h"
#include "test.h"

namespace infini {

namespace {
/// Data an initializer would arrive with, alongside the shape value the pass
/// reads. A constant the fold leaves in place is read by a kernel, which reads
/// data and not the shape value, so leaving the data out only ever looked right
/// where the pool happened to hold the wanted number already.
void fillFromShapeValue(const Tensor &t) {
    t->dataMalloc();
    t->copyin(*t->getShapeValue());
    t->setWeight();
}

/// A rank-one integer constant, as an initializer of a shape subgraph is.
Tensor constantOf(const Graph &g, const vector<int64_t> &values) {
    auto t =
        g->addTensor(Shape{static_cast<int>(values.size())}, DataType::Int64);
    t->setShapeValue(values);
    fillFromShapeValue(t);
    return t;
}

/// A single integer with no dimensions at all. An exporter indexes `Shape`
/// with one of these, so that the gather yields the axis as a number and the
/// `Unsqueeze` after it makes a one-element list.
Tensor scalarOf(const Graph &g, int64_t value) {
    auto t = g->addTensor(Shape{}, DataType::Int64);
    t->setShapeValue(vector<int64_t>{value});
    fillFromShapeValue(t);
    return t;
}
} // namespace

/// The mask says which elements of a shape value cannot change. A dimension
/// the model declared fixed cannot, because `validateShapeChange` rejects any
/// attempt to; one declared dynamic can.
TEST(ShapeFold, MaskFollowsDeclaredDimensions) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 3, 224, 224}, DataType::Float32);
    input->setDimDescs({{true, "batch"}, {false, ""}, {false, ""}, {true, ""}});
    auto shape = g->addOp<ShapeObj>(input, nullptr);

    const auto &value = shape->getOutput()->getShapeValue();
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(*value, (vector<int64_t>{1, 3, 224, 224}));
    EXPECT_FALSE(shape->getOutput()->isShapeValueFixed(0));
    EXPECT_TRUE(shape->getOutput()->isShapeValueFixed(1));
    EXPECT_TRUE(shape->getOutput()->isShapeValueFixed(2));
    EXPECT_FALSE(shape->getOutput()->isShapeValueFixed(3));
    EXPECT_FALSE(shape->getOutput()->isShapeValueWhollyFixed());
}

/// A tensor that never declared its dimensions keeps every one of them
/// replaceable, so nothing about it is settled and nothing may be folded.
TEST(ShapeFold, UndeclaredDimensionsAreNotFixed) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 3}, DataType::Float32);
    auto shape = g->addOp<ShapeObj>(input, nullptr);

    EXPECT_FALSE(shape->getOutput()->isShapeValueFixed(0));
    EXPECT_FALSE(shape->getOutput()->isShapeValueFixed(1));
    EXPECT_EQ(g->foldFixedShapeSubgraph(), 0u);
}

/// Gathering a settled dimension gives a settled element; gathering one that
/// moves does not. This is the case an exporter produces: it reads every axis
/// off `Shape`, including the ones the model fixed.
TEST(ShapeFold, GatherIsAsFixedAsWhatItPicked) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 3, 8, 8}, DataType::Float32);
    input->setDimDescs(
        {{true, "batch"}, {false, ""}, {true, "height"}, {true, "width"}});
    auto shape = g->addOp<ShapeObj>(input, nullptr);
    auto channels =
        g->addOp<GatherObj>(shape->getOutput(), constantOf(g, {1}), nullptr, 0);
    auto batch =
        g->addOp<GatherObj>(shape->getOutput(), constantOf(g, {0}), nullptr, 0);

    EXPECT_TRUE(channels->getOutput()->isShapeValueWhollyFixed());
    EXPECT_FALSE(batch->getOutput()->isShapeValueWhollyFixed());

    // Only the settled branch goes. The one reading the batch size has to be
    // there to read it again when the batch size changes.
    const auto before = g->getOperators().size();
    EXPECT_EQ(g->foldFixedShapeSubgraph(), 1u);
    EXPECT_EQ(g->getOperators().size(), before - 1);
    EXPECT_TRUE(g->checkValid());
}

/// A whole settled chain folds, not merely its first step, because the mask
/// carries along it during shape inference.
TEST(ShapeFold, SettledChainFoldsThroughout) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 3, 8, 8}, DataType::Float32);
    input->setDimDescs(
        {{true, "batch"}, {false, ""}, {false, ""}, {false, ""}});
    auto shape = g->addOp<ShapeObj>(input, nullptr);
    auto height =
        g->addOp<GatherObj>(shape->getOutput(), scalarOf(g, 2), nullptr, 0);
    auto lifted =
        g->addOp<UnsqueezeObj>(height->getOutput(), nullptr, Shape{0});
    auto joined = g->addOp<ConcatObj>(
        TensorVec{lifted->getOutput(), constantOf(g, {4})}, nullptr, 0);

    EXPECT_TRUE(joined->getOutput()->isShapeValueWhollyFixed());
    EXPECT_EQ(*joined->getOutput()->getShapeValue(), (vector<int64_t>{8, 4}));

    EXPECT_EQ(g->getOperators().size(), 4u);
    EXPECT_EQ(g->foldFixedShapeSubgraph(), 4u);
    EXPECT_EQ(g->getOperators().size(), 0u);
    EXPECT_TRUE(g->checkValid());

    // The result is still there to be read, and now stands on its own.
    EXPECT_EQ(*joined->getOutput()->getShapeValue(), (vector<int64_t>{8, 4}));
    EXPECT_EQ(joined->getOutput()->getSource(), nullptr);
}

/// A `Concat` joining a settled dimension with one that moves keeps running,
/// and its kernel reads its inputs, so the folded one must hold real bytes.
TEST(ShapeFold, FoldedTensorCarriesItsDataForALiveConsumer) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 3, 8, 8}, DataType::Float32);
    input->setDimDescs(
        {{true, "batch"}, {false, ""}, {false, ""}, {false, ""}});
    auto shape = g->addOp<ShapeObj>(input, nullptr);
    auto batch =
        g->addOp<GatherObj>(shape->getOutput(), scalarOf(g, 0), nullptr, 0);
    auto height =
        g->addOp<GatherObj>(shape->getOutput(), scalarOf(g, 2), nullptr, 0);
    auto liveBranch =
        g->addOp<UnsqueezeObj>(batch->getOutput(), nullptr, Shape{0});
    auto settledBranch =
        g->addOp<UnsqueezeObj>(height->getOutput(), nullptr, Shape{0});
    auto joined = g->addOp<ConcatObj>(
        TensorVec{liveBranch->getOutput(), settledBranch->getOutput()}, nullptr,
        0);

    // The join reads the batch size, so it is not settled and stays.
    EXPECT_FALSE(joined->getOutput()->isShapeValueWhollyFixed());
    EXPECT_EQ(g->foldFixedShapeSubgraph(), 2u);
    EXPECT_TRUE(g->checkValid());

    g->dataMalloc();
    runtime->run(g);
    EXPECT_EQ(g->cloneTensor(joined->getOutput())->copyout<int64_t>(),
              (vector<int64_t>{1, 8}));
}

/// A folded constant has to outlive every allocation that follows, because
/// nothing runs to fill it again. Allocation plans storage for a tensor with no
/// producer and hands the offset back once the last reader has run, which is
/// right for something refilled from outside each run and wrong for this.
///
/// Whether the offset is then reused at all decides whether the fault shows, so
/// the graph is built to leave no choice: folding the settled branch leaves one
/// constant of eight elements, and the operator planned after the join asks for
/// exactly the sixty-four bytes the join has just released. Every other hole
/// here is smaller, so any allocator must take that one.
TEST(ShapeFold, FoldedConstantKeepsItsStorage) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 16}, DataType::Float32);
    input->setDimDescs({{true, "batch"}, {false, ""}});
    auto shape = g->addOp<ShapeObj>(input, nullptr);
    auto batch =
        g->addOp<GatherObj>(shape->getOutput(), scalarOf(g, 0), nullptr, 0);
    auto width =
        g->addOp<GatherObj>(shape->getOutput(), scalarOf(g, 1), nullptr, 0);

    // Eight copies of the settled width, so that what folding leaves is a
    // constant large enough to be the only hole worth reusing.
    TensorVec settledParts;
    for (int i = 0; i < 8; ++i) {
        settledParts.push_back(
            g->addOp<UnsqueezeObj>(width->getOutput(), nullptr, Shape{0})
                ->getOutput());
    }
    auto settledList = g->addOp<ConcatObj>(settledParts, nullptr, 0);
    auto liveOne =
        g->addOp<UnsqueezeObj>(batch->getOutput(), nullptr, Shape{0});
    auto joined = g->addOp<ConcatObj>(
        TensorVec{liveOne->getOutput(), settledList->getOutput()}, nullptr, 0);

    // Added last so that it is planned after the join has released the
    // constant, and shaped so that it asks for just as much.
    auto big = g->addOp<ReluObj>(input, nullptr);
    ASSERT_EQ(big->getOutput()->getBytes(),
              settledList->getOutput()->getBytes());

    // The settled gather, its eight unsqueezes and their join.
    ASSERT_EQ(g->foldFixedShapeSubgraph(), 10u);
    ASSERT_TRUE(g->checkValid());

    const vector<int64_t> expected{1, 16, 16, 16, 16, 16, 16, 16, 16};
    g->dataMalloc();
    runtime->run(g);
    EXPECT_EQ(g->cloneTensor(joined->getOutput())->copyout<int64_t>(),
              expected);

    // The run above is what hands the constant's storage to another tensor, so
    // it is the second run that reads what became of it.
    runtime->run(g);
    EXPECT_EQ(g->cloneTensor(joined->getOutput())->copyout<int64_t>(),
              expected);

    // A new batch size replans every offset, which the folded width must also
    // survive, twice over for the same reason.
    input->setShape({4, 16});
    g->shape_infer();
    g->dataMalloc();
    const vector<int64_t> reshaped{4, 16, 16, 16, 16, 16, 16, 16, 16};
    runtime->run(g);
    EXPECT_EQ(g->cloneTensor(joined->getOutput())->copyout<int64_t>(),
              reshaped);
    runtime->run(g);
    EXPECT_EQ(g->cloneTensor(joined->getOutput())->copyout<int64_t>(),
              reshaped);
}

/// A settled result asked for as an output of the graph keeps what produces it.
/// The numbers in it would be right either way, but the graph promises that
/// each tensor it holds is either produced by an operator or given from
/// outside, and an output with neither would break that promise -- as
/// `checkValid` here insists.
TEST(ShapeFold, GraphOutputKeepsItsProducer) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{2, 3}, DataType::Float32);
    input->setDimDescs({{false, ""}, {false, ""}});
    auto shape = g->addOp<ShapeObj>(input, nullptr);
    shape->getOutput()->setOutput();

    EXPECT_EQ(g->foldFixedShapeSubgraph(), 0u);
    EXPECT_EQ(g->getOperators().size(), 1u);
    EXPECT_TRUE(g->checkValid());
    const auto &tensors = g->getTensors();
    EXPECT_NE(std::find(tensors.begin(), tensors.end(), shape->getOutput()),
              tensors.end());
    EXPECT_EQ(*shape->getOutput()->getShapeValue(), (vector<int64_t>{2, 3}));
}

/// Folding must not answer differently the second time, so that running the
/// pass again on an already folded graph is safe.
TEST(ShapeFold, FoldingTwiceChangesNothingFurther) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{2, 3}, DataType::Float32);
    input->setDimDescs({{false, ""}, {false, ""}});
    g->addOp<ShapeObj>(input, nullptr);

    EXPECT_EQ(g->foldFixedShapeSubgraph(), 1u);
    EXPECT_EQ(g->foldFixedShapeSubgraph(), 0u);
    EXPECT_TRUE(g->checkValid());
}

/// A shape computation works with the dimensions it read, not only with the
/// dimensions themselves: an exporter writes `d // heads` as a division on the
/// number `Shape` produced. The answer is as settled as both of the numbers it
/// came from, so a fixed dimension over a constant is settled.
TEST(ShapeFold, ArithmeticOnDimensionsCarriesItsValue) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 7, 32}, DataType::Float32);
    input->setDimDescs({{true, "batch"}, {true, "seq"}, {false, ""}});
    auto shape = g->addOp<ShapeObj>(input, nullptr);
    auto width =
        g->addOp<GatherObj>(shape->getOutput(), scalarOf(g, 2), nullptr, 0);
    auto perHead =
        g->addOp<DivObj>(width->getOutput(), scalarOf(g, 4), nullptr);

    const auto &value = perHead->getOutput()->getShapeValue();
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(*value, (vector<int64_t>{8}));
    EXPECT_TRUE(perHead->getOutput()->isShapeValueWhollyFixed());
}

/// The same division reading a dimension that moves. Its answer moves with it,
/// so nothing about it is settled and it must keep running.
TEST(ShapeFold, ArithmeticIsUnsettledWhenEitherSideIs) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 7, 32}, DataType::Float32);
    input->setDimDescs({{true, "batch"}, {true, "seq"}, {false, ""}});
    auto shape = g->addOp<ShapeObj>(input, nullptr);
    auto seq =
        g->addOp<GatherObj>(shape->getOutput(), scalarOf(g, 1), nullptr, 0);
    auto doubled = g->addOp<MulObj>(seq->getOutput(), scalarOf(g, 2), nullptr);

    const auto &value = doubled->getOutput()->getShapeValue();
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(*value, (vector<int64_t>{14}));
    // The number is known for the shape the graph currently has, and is not
    // the same under every shape it may be given.
    EXPECT_FALSE(doubled->getOutput()->isShapeValueFixed(0));
    EXPECT_EQ(g->foldFixedShapeSubgraph(), 0u);
    EXPECT_TRUE(g->checkValid());
}

/// A division with nothing to divide by. There is no number to record, so none
/// is recorded and the operator is left to run as it always did.
TEST(ShapeFold, DivisionByZeroYieldsNoValue) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 7, 32}, DataType::Float32);
    input->setDimDescs({{true, "batch"}, {true, "seq"}, {false, ""}});
    auto shape = g->addOp<ShapeObj>(input, nullptr);
    auto width =
        g->addOp<GatherObj>(shape->getOutput(), scalarOf(g, 2), nullptr, 0);
    auto broken = g->addOp<DivObj>(width->getOutput(), scalarOf(g, 0), nullptr);

    EXPECT_FALSE(broken->getOutput()->getShapeValue().has_value());
    EXPECT_FALSE(broken->getOutput()->isShapeValueWhollyFixed());
    EXPECT_TRUE(g->checkValid());
}

/// A copy hands on both the numbers and how settled they are. An exporter puts
/// one in the middle of a shape subgraph whenever it casts a value to the type
/// it already has, and a chain broken there would leave the `Reshape` below it
/// without a target.
TEST(ShapeFold, CopyPassesShapeValueThrough) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 3, 224, 224}, DataType::Float32);
    input->setDimDescs(
        {{true, "batch"}, {false, ""}, {false, ""}, {false, ""}});
    auto shape = g->addOp<ShapeObj>(input, nullptr);
    auto height =
        g->addOp<GatherObj>(shape->getOutput(), scalarOf(g, 2), nullptr, 0);
    auto copied = g->addOp<IdentityObj>(height->getOutput(), nullptr);

    const auto &value = copied->getOutput()->getShapeValue();
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(*value, (vector<int64_t>{224}));
    EXPECT_TRUE(copied->getOutput()->isShapeValueWhollyFixed());
}

/// The fold takes the arithmetic with it. `32 / 4` is the same eight under
/// every shape the graph may be given, so nothing needs to work it out twice.
TEST(ShapeFold, SettledArithmeticFolds) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 7, 32}, DataType::Float32);
    input->setDimDescs({{true, "batch"}, {true, "seq"}, {false, ""}});
    auto shape = g->addOp<ShapeObj>(input, nullptr);
    auto width =
        g->addOp<GatherObj>(shape->getOutput(), scalarOf(g, 2), nullptr, 0);
    auto perHead =
        g->addOp<DivObj>(width->getOutput(), scalarOf(g, 4), nullptr);
    auto lifted =
        g->addOp<UnsqueezeObj>(perHead->getOutput(), nullptr, vector<int>{0});
    lifted->getOutput()->setOutput();

    // The gather, the division, and the `Shape` left with nobody to read it.
    // The unsqueeze is what the graph was asked for, so it keeps its place and
    // reads the eight the division left behind.
    EXPECT_EQ(g->foldFixedShapeSubgraph(), 3u);
    EXPECT_EQ(g->getOperators().size(), 1u);
    EXPECT_TRUE(g->checkValid());
    EXPECT_EQ(perHead->getOutput()->copyout<int64_t>(), (vector<int64_t>{8}));
}

} // namespace infini
