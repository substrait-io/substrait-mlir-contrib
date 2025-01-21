//===-- Import.cpp - Import protobuf to Substrait dialect -------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "substrait-mlir/Target/SubstraitPB/Import.h"

#include "ProtobufUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OwningOpRef.h"
#include "substrait-mlir/Dialect/Substrait/IR/Substrait.h"
#include "substrait-mlir/Target/SubstraitPB/Options.h"

#include <google/protobuf/descriptor.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/json_util.h>
#include <substrait/proto/algebra.pb.h>
#include <substrait/proto/extensions/extensions.pb.h>
#include <substrait/proto/plan.pb.h>
#include <substrait/proto/type.pb.h>

using namespace mlir;
using namespace mlir::substrait;
using namespace ::substrait;
using namespace ::substrait::proto;

namespace pb = google::protobuf;

namespace {

// Forward declaration for the import function of the given message type.
//
// We need one such function for most message types that we want to import. The
// forward declarations are necessary such all import functions are available
// for the definitions indepedently of the order of these definitions. The
// message type passed to the function (specified by `MESSAGE_TYPE`) may be
// different than the one it is responsible for: often the target op type
// (specified by `OP_TYPE`) depends on a nested field value (such as `oneof`)
// but the import logic needs the whole context; the message that is passed in
// is the most deeply nested message that provides the whole context.
#define DECLARE_IMPORT_FUNC(MESSAGE_TYPE, ARG_TYPE, OP_TYPE)                   \
  static FailureOr<OP_TYPE> import##MESSAGE_TYPE(ImplicitLocOpBuilder builder, \
                                                 const ARG_TYPE &message);

DECLARE_IMPORT_FUNC(CrossRel, Rel, CrossOp)
DECLARE_IMPORT_FUNC(FetchRel, Rel, FetchOp)
DECLARE_IMPORT_FUNC(FilterRel, Rel, FilterOp)
DECLARE_IMPORT_FUNC(SetRel, Rel, SetOp)
DECLARE_IMPORT_FUNC(Expression, Expression, ExpressionOpInterface)
DECLARE_IMPORT_FUNC(FieldReference, Expression::FieldReference,
                    FieldReferenceOp)
DECLARE_IMPORT_FUNC(JoinRel, Rel, JoinOp)
DECLARE_IMPORT_FUNC(Literal, Expression::Literal, LiteralOp)
DECLARE_IMPORT_FUNC(NamedTable, Rel, NamedTableOp)
DECLARE_IMPORT_FUNC(Plan, Plan, PlanOp)
DECLARE_IMPORT_FUNC(PlanRel, PlanRel, PlanRelOp)
DECLARE_IMPORT_FUNC(ProjectRel, Rel, ProjectOp)
DECLARE_IMPORT_FUNC(ReadRel, Rel, RelOpInterface)
DECLARE_IMPORT_FUNC(Rel, Rel, RelOpInterface)
DECLARE_IMPORT_FUNC(ScalarFunction, Expression::ScalarFunction, CallOp)

// Helpers to build symbol names from anchors deterministically. This allows
// to reate symbol references from anchors without look-up structure. Also,
// the format is exploited by the export logic to recover the original anchor
// values of (unmodified) imported plans.

/// Builds a deterministic symbol name for an URI with the given anchor.
static std::string buildUriSymName(int32_t anchor) {
  return ("extension_uri." + Twine(anchor)).str();
}

/// Builds a deterministic symbol name for a function with the given anchor.
static std::string buildFuncSymName(int32_t anchor) {
  return ("extension_function." + Twine(anchor)).str();
}

/// Builds a deterministic symbol name for a type with the given anchor.
static std::string buildTypeSymName(int32_t anchor) {
  return ("extension_type." + Twine(anchor)).str();
}

/// Builds a deterministic symbol name for a type variation with the given
/// anchor.
static std::string buildTypeVarSymName(int32_t anchor) {
  return ("extension_type_variation." + Twine(anchor)).str();
}

static mlir::FailureOr<mlir::Type> importType(MLIRContext *context,
                                              const proto::Type &type) {

  proto::Type::KindCase kindCase = type.kind_case();
  switch (kindCase) {
  case proto::Type::kBool:
    return IntegerType::get(context, 1, IntegerType::Signed);
  case proto::Type::kI8:
    return IntegerType::get(context, 8, IntegerType::Signed);
  case proto::Type::kI16:
    return IntegerType::get(context, 16, IntegerType::Signed);
  case proto::Type::kI32:
    return IntegerType::get(context, 32, IntegerType::Signed);
  case proto::Type::kI64:
    return IntegerType::get(context, 64, IntegerType::Signed);
  case proto::Type::kFp32:
    return FloatType::getF32(context);
  case proto::Type::kFp64:
    return FloatType::getF64(context);
  case proto::Type::kString:
    return StringType::get(context);
  case proto::Type::kBinary:
    return BinaryType::get(context);
  case proto::Type::kTimestamp:
    return TimestampType::get(context);
  case proto::Type::kTimestampTz:
    return TimestampTzType::get(context);
  case proto::Type::kDate:
    return DateType::get(context);
  case proto::Type::kStruct: {
    const proto::Type::Struct &structType = type.struct_();
    llvm::SmallVector<mlir::Type> fieldTypes;
    fieldTypes.reserve(structType.types_size());
    for (const proto::Type &fieldType : structType.types()) {
      FailureOr<mlir::Type> mlirFieldType = importType(context, fieldType);
      if (failed(mlirFieldType))
        return failure();
      fieldTypes.push_back(mlirFieldType.value());
    }
    return TupleType::get(context, fieldTypes);
  }
    // TODO(ingomueller): Support more types.
  default: {
    auto loc = UnknownLoc::get(context);
    const pb::FieldDescriptor *desc =
        proto::Type::GetDescriptor()->FindFieldByNumber(kindCase);
    assert(desc && "could not get field descriptor");
    return emitError(loc) << "could not import unsupported type "
                          << desc->name();
  }
  }
}

static mlir::FailureOr<CrossOp> importCrossRel(ImplicitLocOpBuilder builder,
                                               const Rel &message) {
  const CrossRel &crossRel = message.cross();

  // Import left and right inputs.
  const Rel &leftRel = crossRel.left();
  const Rel &rightRel = crossRel.right();

  mlir::FailureOr<RelOpInterface> leftOp = importRel(builder, leftRel);
  mlir::FailureOr<RelOpInterface> rightOp = importRel(builder, rightRel);

  if (failed(leftOp) || failed(rightOp))
    return failure();

  // Build `CrossOp`.
  Value leftVal = leftOp.value()->getResult(0);
  Value rightVal = rightOp.value()->getResult(0);

  return builder.create<CrossOp>(leftVal, rightVal);
}

static mlir::FailureOr<SetOp> importSetRel(ImplicitLocOpBuilder builder,
                                           const Rel &message) {
  const SetRel &setRel = message.set();

  // Import inputs
  const google::protobuf::RepeatedPtrField<Rel> &inputsRel = setRel.inputs();

  // Build `SetOp`.
  llvm::SmallVector<Value> inputsVal;

  for (const Rel &inputRel : inputsRel) {
    mlir::FailureOr<RelOpInterface> inputOp = importRel(builder, inputRel);
    if (failed(inputOp))
      return failure();
    inputsVal.push_back(inputOp.value()->getResult(0));
  }

  std::optional<SetOpKind> kind = static_cast<::SetOpKind>(setRel.op());

  // Check for unsupported set operations.
  if (!kind)
    return mlir::emitError(builder.getLoc(), "unexpected 'operation' found");

  return builder.create<SetOp>(inputsVal, *kind);
}

static mlir::FailureOr<ExpressionOpInterface>
importExpression(ImplicitLocOpBuilder builder, const Expression &message) {
  MLIRContext *context = builder.getContext();
  Location loc = UnknownLoc::get(context);

  Expression::RexTypeCase rex_type = message.rex_type_case();
  switch (rex_type) {
  case Expression::kLiteral:
    return importLiteral(builder, message.literal());
  case Expression::kSelection:
    return importFieldReference(builder, message.selection());
  case Expression::kScalarFunction:
    return importScalarFunction(builder, message.scalar_function());
  default: {
    const pb::FieldDescriptor *desc =
        Expression::GetDescriptor()->FindFieldByNumber(rex_type);
    return emitError(loc) << Twine("unsupported Expression type: ") +
                                 desc->name();
  }
  }
}

static mlir::FailureOr<FieldReferenceOp>
importFieldReference(ImplicitLocOpBuilder builder,
                     const Expression::FieldReference &message) {
  using ReferenceSegment = Expression::ReferenceSegment;

  MLIRContext *context = builder.getContext();
  Location loc = UnknownLoc::get(context);

  // Emit error on unsupported cases.
  // TODO(ingomueller): support more cases.
  if (!message.has_direct_reference())
    return emitError(loc) << "only direct reference supported";

  // Traverse list to extract indices.
  llvm::SmallVector<int64_t> indices;
  const ReferenceSegment *currentSegment = &message.direct_reference();
  while (true) {
    if (!currentSegment->has_struct_field())
      return emitError(loc) << "only struct fields supported";

    const ReferenceSegment::StructField &structField =
        currentSegment->struct_field();
    indices.push_back(structField.field());

    // Continue in linked list or end traversal.
    if (!structField.has_child())
      break;
    currentSegment = &structField.child();
  }

  // Get input value.
  Value container;
  if (message.has_root_reference()) {
    // For the `root_reference` case, that's the current block argument.
    mlir::Block::BlockArgListType blockArgs =
        builder.getInsertionBlock()->getArguments();
    assert(blockArgs.size() == 1 && "expected a single block argument");
    container = blockArgs.front();
  } else if (message.has_expression()) {
    // For the `expression` case, recursively import the expression.
    FailureOr<ExpressionOpInterface> maybeContainer =
        importExpression(builder, message.expression());
    if (failed(maybeContainer))
      return failure();
    container = maybeContainer.value()->getResult(0);
  } else {
    // For the `outer_reference` case, we need to refer to an argument of some
    // outer-level block.
    // TODO(ingomueller): support outer references.
    assert(message.has_outer_reference() && "unexpected 'root_type` case");
    return emitError(loc) << "outer references not supported";
  }

  // Build and return the op.
  return builder.create<FieldReferenceOp>(container, indices);
}

static mlir::FailureOr<JoinOp> importJoinRel(ImplicitLocOpBuilder builder,
                                             const Rel &message) {
  const JoinRel &joinRel = message.join();

  // Import left and right inputs.
  const Rel &leftRel = joinRel.left();
  const Rel &rightRel = joinRel.right();

  mlir::FailureOr<RelOpInterface> leftOp = importRel(builder, leftRel);
  mlir::FailureOr<RelOpInterface> rightOp = importRel(builder, rightRel);

  if (failed(leftOp) || failed(rightOp))
    return failure();

  // Build `JoinOp`.
  Value leftVal = leftOp.value()->getResult(0);
  Value rightVal = rightOp.value()->getResult(0);

  std::optional<JoinTypeKind> join_type =
      static_cast<::JoinTypeKind>(joinRel.type());

  // Check for unsupported set operations.
  if (!join_type)
    return mlir::emitError(builder.getLoc(), "unexpected 'operation' found");

  return builder.create<JoinOp>(leftVal, rightVal, *join_type);
}

static mlir::FailureOr<LiteralOp>
importLiteral(ImplicitLocOpBuilder builder,
              const Expression::Literal &message) {
  MLIRContext *context = builder.getContext();
  Location loc = UnknownLoc::get(context);

  Expression::Literal::LiteralTypeCase literalType =
      message.literal_type_case();
  switch (literalType) {
  case Expression::Literal::LiteralTypeCase::kBoolean: {
    auto attr = IntegerAttr::get(
        IntegerType::get(context, 1, IntegerType::Signed), message.boolean());
    return builder.create<LiteralOp>(attr);
  }
  case Expression::Literal::LiteralTypeCase::kI8: {
    auto attr = IntegerAttr::get(
        IntegerType::get(context, 8, IntegerType::Signed), message.i8());
    return builder.create<LiteralOp>(attr);
  }
  case Expression::Literal::LiteralTypeCase::kI16: {
    auto attr = IntegerAttr::get(
        IntegerType::get(context, 16, IntegerType::Signed), message.i16());
    return builder.create<LiteralOp>(attr);
  }
  case Expression::Literal::LiteralTypeCase::kI32: {
    auto attr = IntegerAttr::get(
        IntegerType::get(context, 32, IntegerType::Signed), message.i32());
    return builder.create<LiteralOp>(attr);
  }
  case Expression::Literal::LiteralTypeCase::kI64: {
    auto attr = IntegerAttr::get(
        IntegerType::get(context, 64, IntegerType::Signed), message.i64());
    return builder.create<LiteralOp>(attr);
  }
  case Expression::Literal::LiteralTypeCase::kFp32: {
    auto attr = FloatAttr::get(FloatType::getF32(context), message.fp32());
    return builder.create<LiteralOp>(attr);
  }
  case Expression::Literal::LiteralTypeCase::kFp64: {
    auto attr = FloatAttr::get(FloatType::getF64(context), message.fp64());
    return builder.create<LiteralOp>(attr);
  }
  case Expression::Literal::LiteralTypeCase::kString: {
    auto attr = StringAttr::get(message.string(), StringType::get(context));
    return builder.create<LiteralOp>(attr);
  }
  case Expression::Literal::LiteralTypeCase::kBinary: {
    auto attr = StringAttr::get(message.binary(), BinaryType::get(context));
    return builder.create<LiteralOp>(attr);
  }
  case Expression::Literal::LiteralTypeCase::kTimestamp: {
    auto attr = TimestampAttr::get(context, message.timestamp());
    return builder.create<LiteralOp>(attr);
  }
  case Expression::Literal::LiteralTypeCase::kTimestampTz: {
    auto attr = TimestampTzAttr::get(context, message.timestamp_tz());
    return builder.create<LiteralOp>(attr);
  }
  case Expression::Literal::LiteralTypeCase::kDate: {
    auto attr = DateAttr::get(context, message.date());
    return builder.create<LiteralOp>(attr);
  }
  // TODO(ingomueller): Support more types.
  default: {
    const pb::FieldDescriptor *desc =
        Expression::Literal::GetDescriptor()->FindFieldByNumber(literalType);
    return emitError(loc) << Twine("unsupported Literal type: ") + desc->name();
  }
  }
}

static mlir::FailureOr<FetchOp> importFetchRel(ImplicitLocOpBuilder builder,
                                               const Rel &message) {
  const FetchRel &fetchRel = message.fetch();

  // Import input.
  const Rel &inputRel = fetchRel.input();
  mlir::FailureOr<RelOpInterface> inputOp = importRel(builder, inputRel);

  // Build `FetchOp`.
  Value inputVal = inputOp.value()->getResult(0);
  return builder.create<FetchOp>(inputVal, fetchRel.offset(), fetchRel.count());
}

static mlir::FailureOr<FilterOp> importFilterRel(ImplicitLocOpBuilder builder,
                                                 const Rel &message) {
  const FilterRel &filterRel = message.filter();

  // Import input op.
  const Rel &inputRel = filterRel.input();
  mlir::FailureOr<RelOpInterface> inputOp = importRel(builder, inputRel);
  if (failed(inputOp))
    return failure();

  // Create filter op.
  auto filterOp = builder.create<FilterOp>(inputOp.value()->getResult(0));
  filterOp.getCondition().push_back(new Block);
  Block &conditionBlock = filterOp.getCondition().front();
  conditionBlock.addArgument(filterOp.getResult().getType(),
                             filterOp->getLoc());

  // Create condition region.
  const Expression &expression = filterRel.condition();
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&conditionBlock);

    FailureOr<ExpressionOpInterface> conditionOp =
        importExpression(builder, expression);
    if (failed(conditionOp))
      return failure();

    builder.create<YieldOp>(conditionOp.value()->getResult(0));
  }

  return filterOp;
}

static mlir::FailureOr<NamedTableOp>
importNamedTable(ImplicitLocOpBuilder builder, const Rel &message) {
  const ReadRel &readRel = message.read();
  const ReadRel::NamedTable &namedTable = readRel.named_table();
  MLIRContext *context = builder.getContext();

  // Assemble table name.
  llvm::SmallVector<FlatSymbolRefAttr> tableNameRefs;
  tableNameRefs.reserve(namedTable.names_size());
  for (const std::string &name : namedTable.names()) {
    auto attr = FlatSymbolRefAttr::get(context, name);
    tableNameRefs.push_back(attr);
  }
  llvm::ArrayRef<FlatSymbolRefAttr> tableNameNestedRefs =
      llvm::ArrayRef<FlatSymbolRefAttr>(tableNameRefs).drop_front();
  llvm::StringRef tableNameRootRef = tableNameRefs.front().getValue();
  auto tableName =
      SymbolRefAttr::get(context, tableNameRootRef, tableNameNestedRefs);

  // Assemble field names from schema.
  const NamedStruct &baseSchema = readRel.base_schema();
  llvm::SmallVector<Attribute> fieldNames;
  fieldNames.reserve(baseSchema.names_size());
  for (const std::string &name : baseSchema.names()) {
    auto attr = StringAttr::get(context, name);
    fieldNames.push_back(attr);
  }
  auto fieldNamesAttr = ArrayAttr::get(context, fieldNames);

  // Assemble field names from schema.
  const proto::Type::Struct &struct_ = baseSchema.struct_();
  llvm::SmallVector<mlir::Type> resultTypes;
  resultTypes.reserve(struct_.types_size());
  for (const proto::Type &type : struct_.types()) {
    FailureOr<mlir::Type> mlirType = importType(context, type);
    if (failed(mlirType))
      return failure();
    resultTypes.push_back(mlirType.value());
  }
  auto resultType = TupleType::get(context, resultTypes);

  // Assemble final op.
  auto namedTableOp =
      builder.create<NamedTableOp>(resultType, tableName, fieldNamesAttr);

  return namedTableOp;
}

static FailureOr<PlanOp> importPlan(ImplicitLocOpBuilder builder,
                                    const Plan &message) {
  using extensions::SimpleExtensionDeclaration;
  using extensions::SimpleExtensionURI;
  using ExtensionFunction = SimpleExtensionDeclaration::ExtensionFunction;
  using ExtensionType = SimpleExtensionDeclaration::ExtensionType;
  using ExtensionTypeVariation =
      SimpleExtensionDeclaration::ExtensionTypeVariation;

  MLIRContext *context = builder.getContext();
  Location loc = UnknownLoc::get(context);

  const Version &version = message.version();
  auto planOp = builder.create<PlanOp>(
      version.major_number(), version.minor_number(), version.patch_number(),
      version.git_hash(), version.producer());
  planOp.getBody().push_back(new Block());

  OpBuilder::InsertionGuard insertGuard(builder);
  builder.setInsertionPointToEnd(&planOp.getBody().front());

  // Import `extension_uris` creating symbol names deterministically.
  for (const SimpleExtensionURI &extUri : message.extension_uris()) {
    int32_t anchor = extUri.extension_uri_anchor();
    StringRef uri = extUri.uri();
    std::string symName = buildUriSymName(anchor);
    builder.create<ExtensionUriOp>(symName, uri);
  }

  // Import `extension`s reconstructing symbol references to URI ops from the
  // corresponding anchors using the same method as above.
  for (const SimpleExtensionDeclaration &ext : message.extensions()) {
    SimpleExtensionDeclaration::MappingTypeCase mappingCase =
        ext.mapping_type_case();
    switch (mappingCase) {
    case SimpleExtensionDeclaration::kExtensionFunction: {
      const ExtensionFunction &func = ext.extension_function();
      int32_t anchor = func.function_anchor();
      int32_t uriRef = func.extension_uri_reference();
      const std::string &funcName = func.name();
      std::string symName = buildFuncSymName(anchor);
      std::string uriSymName = buildUriSymName(uriRef);
      builder.create<ExtensionFunctionOp>(symName, uriSymName, funcName);
      break;
    }
    case SimpleExtensionDeclaration::kExtensionType: {
      const ExtensionType &type = ext.extension_type();
      int32_t anchor = type.type_anchor();
      int32_t uriRef = type.extension_uri_reference();
      const std::string &typeName = type.name();
      std::string symName = buildTypeSymName(anchor);
      std::string uriSymName = buildUriSymName(uriRef);
      builder.create<ExtensionTypeOp>(symName, uriSymName, typeName);
      break;
    }
    case SimpleExtensionDeclaration::kExtensionTypeVariation: {
      const ExtensionTypeVariation &typeVar = ext.extension_type_variation();
      int32_t anchor = typeVar.type_variation_anchor();
      int32_t uriRef = typeVar.extension_uri_reference();
      const std::string &typeVarName = typeVar.name();
      std::string symName = buildTypeVarSymName(anchor);
      std::string uriSymName = buildUriSymName(uriRef);
      builder.create<ExtensionTypeVariationOp>(symName, uriSymName,
                                               typeVarName);
      break;
    }
    default:
      const pb::FieldDescriptor *desc =
          SimpleExtensionDeclaration::GetDescriptor()->FindFieldByNumber(
              mappingCase);
      return emitError(loc)
             << Twine("unsupported SimpleExtensionDeclaration type: ") +
                    desc->name();
    }
  }

  for (const PlanRel &relation : message.relations()) {
    if (failed(importPlanRel(builder, relation)))
      return failure();
  }

  return planOp;
}

static FailureOr<PlanRelOp> importPlanRel(ImplicitLocOpBuilder builder,
                                          const PlanRel &message) {
  MLIRContext *context = builder.getContext();
  Location loc = UnknownLoc::get(context);

  if (!message.has_rel() && !message.has_root()) {
    PlanRel::RelTypeCase relType = message.rel_type_case();
    const pb::FieldDescriptor *desc =
        PlanRel::GetDescriptor()->FindFieldByNumber(relType);
    return emitError(loc) << Twine("unsupported PlanRel type: ") + desc->name();
  }

  // Create new `PlanRelOp`.
  auto planRelOp = builder.create<PlanRelOp>();
  planRelOp.getBody().push_back(new Block());
  Block *block = &planRelOp.getBody().front();

  // Handle `Rel` and `RelRoot` separately.
  const Rel *rel;
  if (message.has_rel())
    rel = &message.rel();
  else {
    const RelRoot &root = message.root();
    rel = &root.input();

    // Extract names.
    SmallVector<std::string> names(root.names().begin(), root.names().end());
    SmallVector<llvm::StringRef> nameAttrs(names.begin(), names.end());
    ArrayAttr namesAttr = builder.getStrArrayAttr(nameAttrs);
    planRelOp.setFieldNamesAttr(namesAttr);
  }

  // Import body of `PlanRelOp`.
  OpBuilder::InsertionGuard insertGuard(builder);
  builder.setInsertionPointToEnd(block);
  mlir::FailureOr<Operation *> rootRel = importRel(builder, *rel);
  if (failed(rootRel))
    return failure();

  builder.setInsertionPointToEnd(block);
  builder.create<YieldOp>(rootRel.value()->getResult(0));

  return planRelOp;
}

static mlir::FailureOr<ProjectOp> importProjectRel(ImplicitLocOpBuilder builder,
                                                   const Rel &message) {
  const ProjectRel &projectRel = message.project();

  // Import input op.
  const Rel &inputRel = projectRel.input();
  mlir::FailureOr<RelOpInterface> inputOp = importRel(builder, inputRel);
  if (failed(inputOp))
    return failure();

  // Create `expressions` block.
  auto conditionBlock = std::make_unique<Block>();
  auto inputTupleType =
      cast<TupleType>(inputOp.value()->getResult(0).getType());
  conditionBlock->addArgument(inputTupleType, inputOp->getLoc());

  // Fill `expressions` block with expression trees.
  YieldOp yieldOp;
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(conditionBlock.get());

    SmallVector<Value> values;
    values.reserve(projectRel.expressions_size());
    for (const Expression &expression : projectRel.expressions()) {
      // Import expression tree recursively.
      FailureOr<ExpressionOpInterface> rootExprOp =
          importExpression(builder, expression);
      if (failed(rootExprOp))
        return failure();
      values.push_back(rootExprOp.value()->getResult(0));
    }

    // Create final `yield` op with root expression values.
    yieldOp = builder.create<YieldOp>(values);
  }

  // Compute output type.
  SmallVector<mlir::Type> resultFieldTypes;
  resultFieldTypes.reserve(inputTupleType.size() + yieldOp->getNumOperands());
  append_range(resultFieldTypes, inputTupleType);
  append_range(resultFieldTypes, yieldOp->getOperandTypes());
  auto resultType = TupleType::get(builder.getContext(), resultFieldTypes);

  // Create `project` op.
  auto projectOp =
      builder.create<ProjectOp>(resultType, inputOp.value()->getResult(0));
  projectOp.getExpressions().push_back(conditionBlock.release());

  return projectOp;
}

static mlir::FailureOr<RelOpInterface>
importReadRel(ImplicitLocOpBuilder builder, const Rel &message) {
  MLIRContext *context = builder.getContext();
  Location loc = UnknownLoc::get(context);

  const ReadRel &readRel = message.read();
  ReadRel::ReadTypeCase readType = readRel.read_type_case();
  switch (readType) {
  case ReadRel::ReadTypeCase::kNamedTable: {
    return importNamedTable(builder, message);
  }
  default:
    const pb::FieldDescriptor *desc =
        ReadRel::GetDescriptor()->FindFieldByNumber(readType);
    return emitError(loc) << Twine("unsupported ReadRel type: ") + desc->name();
  }
}

static mlir::FailureOr<RelOpInterface> importRel(ImplicitLocOpBuilder builder,
                                                 const Rel &message) {
  MLIRContext *context = builder.getContext();
  Location loc = UnknownLoc::get(context);

  // Import rel depending on its type.
  Rel::RelTypeCase relType = message.rel_type_case();
  FailureOr<RelOpInterface> maybeOp;
  switch (relType) {
  case Rel::RelTypeCase::kCross:
    maybeOp = importCrossRel(builder, message);
    break;
  case Rel::RelTypeCase::kFetch:
    maybeOp = importFetchRel(builder, message);
    break;
  case Rel::RelTypeCase::kFilter:
    maybeOp = importFilterRel(builder, message);
    break;
  case Rel::RelTypeCase::kJoin:
    maybeOp = importJoinRel(builder, message);
    break;
  case Rel::RelTypeCase::kProject:
    maybeOp = importProjectRel(builder, message);
    break;
  case Rel::RelTypeCase::kRead:
    maybeOp = importReadRel(builder, message);
    break;
  case Rel::RelTypeCase::kSet:
    maybeOp = importSetRel(builder, message);
    break;
  default:
    const pb::FieldDescriptor *desc =
        Rel::GetDescriptor()->FindFieldByNumber(relType);
    return emitError(loc) << Twine("unsupported Rel type: ") + desc->name();
  }
  if (failed(maybeOp))
    return failure();
  RelOpInterface op = maybeOp.value();

  // Remainder: Import `emit` op if needed.

  // Extract `RelCommon` message.
  FailureOr<const RelCommon *> maybeRelCommon =
      protobuf_utils::getCommon(message, loc);
  if (failed(maybeRelCommon))
    return failure();
  const RelCommon *relCommon = maybeRelCommon.value();

  // For the `direct` case, no further op needs to be created.
  if (relCommon->has_direct())
    return op;
  assert(relCommon->has_emit() && "expected either 'direct' or 'emit'");

  // For the `emit` case, we need to insert an `EmitOp`.
  const proto::RelCommon::Emit &emit = relCommon->emit();
  SmallVector<int64_t> mapping;
  append_range(mapping, emit.output_mapping());
  ArrayAttr mappingAttr = builder.getI64ArrayAttr(mapping);
  auto emitOp = builder.create<EmitOp>(op->getResult(0), mappingAttr);

  return {emitOp};
}

static mlir::FailureOr<CallOp>
importScalarFunction(ImplicitLocOpBuilder builder,
                     const Expression::ScalarFunction &message) {
  MLIRContext *context = builder.getContext();
  Location loc = UnknownLoc::get(context);

  // Import `output_type`.
  const proto::Type &outputType = message.output_type();
  FailureOr<mlir::Type> mlirOutputType = importType(context, outputType);
  if (failed(mlirOutputType))
    return failure();

  // Import `arguments`.
  SmallVector<Value> operands;
  for (const FunctionArgument &arg : message.arguments()) {
    // Error out on unsupported cases.
    // TODO(ingomueller): Support other function argument types.
    if (!arg.has_value()) {
      const pb::FieldDescriptor *desc =
          FunctionArgument::GetDescriptor()->FindFieldByNumber(
              arg.arg_type_case());
      return emitError(loc) << Twine("unsupported arg type: ") + desc->name();
    }

    // Handle `value` case.
    const Expression &value = arg.value();
    FailureOr<ExpressionOpInterface> expression =
        importExpression(builder, value);
    if (failed(expression))
      return failure();
    operands.push_back((*expression)->getResult(0));
  }

  // Import `function_reference` field.
  int32_t anchor = message.function_reference();
  std::string calleeSymName = buildFuncSymName(anchor);

  // Create op.
  auto callOp =
      builder.create<CallOp>(mlirOutputType.value(), calleeSymName, operands);

  return {callOp};
}

} // namespace

namespace mlir {
namespace substrait {

OwningOpRef<ModuleOp>
translateProtobufToSubstrait(llvm::StringRef input, MLIRContext *context,
                             ImportExportOptions options) {
  Location loc = UnknownLoc::get(context);
  auto plan = std::make_unique<Plan>();
  switch (options.serdeFormat) {
  case substrait::SerdeFormat::kText:
    if (!pb::TextFormat::ParseFromString(input.str(), plan.get())) {
      emitError(loc) << "could not parse string as 'Plan' message.";
      return {};
    }
    break;
  case substrait::SerdeFormat::kBinary:
    if (!plan->ParseFromString(input.str())) {
      emitError(loc) << "could not deserialize input as 'Plan' message.";
      return {};
    }
    break;
  case substrait::SerdeFormat::kJson:
  case substrait::SerdeFormat::kPrettyJson: {
    pb::util::Status status =
        pb::util::JsonStringToMessage(input.str(), plan.get());
    if (!status.ok()) {
      emitError(loc) << "could not deserialize JSON as 'Plan' message:\n"
                     << status.message().as_string();
      return {};
    }
  }
  }

  context->loadDialect<SubstraitDialect>();

  ImplicitLocOpBuilder builder(loc, context);
  auto module = builder.create<ModuleOp>(loc);
  auto moduleRef = OwningOpRef<ModuleOp>(module);
  builder.setInsertionPointToEnd(&module.getBodyRegion().back());

  if (failed(importPlan(builder, *plan)))
    return {};

  return moduleRef;
}

} // namespace substrait
} // namespace mlir
