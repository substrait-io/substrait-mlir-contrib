//===-- Substrait.cpp - Substrait dialect -----------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "substrait-mlir/Dialect/Substrait/IR/Substrait.h"

#include "mlir/IR/DialectImplementation.h" // IWYU pragma: keep
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h" // IWYU pragma: keep

using namespace mlir;
using namespace mlir::substrait;

//===----------------------------------------------------------------------===//
// Substrait dialect
//===----------------------------------------------------------------------===//

#include "substrait-mlir/Dialect/Substrait/IR/SubstraitOpsDialect.cpp.inc"

void SubstraitDialect::initialize() {
#define GET_OP_LIST
  addOperations<
#include "substrait-mlir/Dialect/Substrait/IR/SubstraitOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "substrait-mlir/Dialect/Substrait/IR/SubstraitOpsTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "substrait-mlir/Dialect/Substrait/IR/SubstraitOpsAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Free functions
//===----------------------------------------------------------------------===//

namespace mlir::substrait {

Type getAttrType(Attribute attr) {
  if (auto typedAttr = mlir::dyn_cast<TypedAttr>(attr))
    return typedAttr.getType();
  if (auto typedAttr = mlir::dyn_cast<TypeInferableAttrInterface>(attr))
    return typedAttr.getType();
  return Type();
}

} // namespace mlir::substrait

//===----------------------------------------------------------------------===//
// Substrait attributes
//===----------------------------------------------------------------------===//

LogicalResult AdvancedExtensionAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::StringAttr optimization, mlir::StringAttr enhancement) {
  if (optimization && !mlir::isa<AnyType>(optimization.getType()))
    return emitError() << "has 'optimization' attribute of wrong type";
  if (enhancement && !mlir::isa<AnyType>(enhancement.getType()))
    return emitError() << "has 'enhancement' attribute of wrong type";
  return success();
}

//===----------------------------------------------------------------------===//
// Substrait enums
//===----------------------------------------------------------------------===//

#include "substrait-mlir/Dialect/Substrait/IR/SubstraitEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// Substrait interfaces
//===----------------------------------------------------------------------===//

#include "substrait-mlir/Dialect/Substrait/IR/SubstraitAttrInterfaces.cpp.inc"
#include "substrait-mlir/Dialect/Substrait/IR/SubstraitOpInterfaces.cpp.inc"
#include "substrait-mlir/Dialect/Substrait/IR/SubstraitTypeInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Custom Parser and Printer for Substrait
//===----------------------------------------------------------------------===//

bool parseCountAsAll(OpAsmParser &parser, IntegerAttr &count) {
  // `all` keyword (corresponds to `-1`).
  if (!parser.parseOptionalKeyword("all")) {
    count = parser.getBuilder().getI64IntegerAttr(-1);
    return false;
  }

  // Normal integer.
  int64_t result;
  if (!parser.parseInteger(result)) {
    count = parser.getBuilder().getI64IntegerAttr(result);
    return false;
  }

  // Error.
  return true;
}

void printCountAsAll(OpAsmPrinter &printer, Operation *op, IntegerAttr count) {
  if (count.getInt() == -1) {
    printer << "all";
    return;
  }
  // Normal integer.
  printer << count.getValue();
}

//===----------------------------------------------------------------------===//
// Substrait operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "substrait-mlir/Dialect/Substrait/IR/SubstraitOps.cpp.inc"

namespace mlir {
namespace substrait {

/// Implement `SymbolOpInterface`.
::mlir::LogicalResult
CallOp::verifySymbolUses(SymbolTableCollection &symbolTables) {
  if (!symbolTables.lookupNearestSymbolFrom<ExtensionFunctionOp>(
          *this, getCalleeAttr()))
    return emitOpError() << "refers to " << getCalleeAttr()
                         << ", which is not a valid 'extension_function' op";
  return success();
}

LogicalResult
CrossOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc,
                          ValueRange operands, DictionaryAttr attributes,
                          OpaqueProperties properties, RegionRange regions,
                          llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
  Value leftInput = operands[0];
  Value rightInput = operands[1];

  TypeRange leftFieldTypes = cast<TupleType>(leftInput.getType()).getTypes();
  TypeRange rightFieldTypes = cast<TupleType>(rightInput.getType()).getTypes();

  SmallVector<mlir::Type> fieldTypes;
  llvm::append_range(fieldTypes, leftFieldTypes);
  llvm::append_range(fieldTypes, rightFieldTypes);
  auto resultType = TupleType::get(context, fieldTypes);

  inferredReturnTypes = SmallVector<Type>{resultType};

  return success();
}

OpFoldResult EmitOp::fold(FoldAdaptor adaptor) {
  MLIRContext *context = getContext();
  Type i64 = IntegerType::get(context, 64);

  // If the input is also an `emit`, fold it into this op.
  if (auto previousEmit = dyn_cast<EmitOp>(getInput().getDefiningOp())) {
    // Compute new mapping.
    ArrayAttr previousMapping = previousEmit.getMapping();
    SmallVector<Attribute> newMapping;
    newMapping.reserve(getMapping().size());
    for (auto attr : getMapping().getAsRange<IntegerAttr>()) {
      int64_t index = attr.getInt();
      int64_t newIndex = cast<IntegerAttr>(previousMapping[index]).getInt();
      newMapping.push_back(IntegerAttr::get(i64, newIndex));
    }

    // Update this op.
    setMappingAttr(ArrayAttr::get(context, newMapping));
    setOperand(previousEmit.getInput());
    return getResult();
  }

  // Remainder: fold away if the mapping is the identity mapping.

  // Return if the mapping is not the identity mapping.
  int64_t numFields = cast<TupleType>(getInput().getType()).size();
  int64_t numIndices = getMapping().size();
  if (numFields != numIndices)
    return {};
  for (int64_t i = 0; i < numIndices; ++i) {
    auto attr = getMapping()[i];
    int64_t index = cast<IntegerAttr>(attr).getInt();
    if (index != i)
      return {};
  }

  // The `emit` op *has* an identity mapping, so it does not have any effect.
  // Return its input instead.
  return getInput();
}

LogicalResult
EmitOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc,
                         ValueRange operands, DictionaryAttr attributes,
                         OpaqueProperties properties, RegionRange regions,
                         llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
  auto *typedProperties = properties.as<Properties *>();
  if (!loc)
    loc = UnknownLoc::get(context);

  ArrayAttr mapping = typedProperties->getMapping();
  Type inputType = operands[0].getType();
  ArrayRef<Type> inputTypes = mlir::cast<TupleType>(inputType).getTypes();

  // Map input types to output types.
  SmallVector<Type> outputTypes;
  outputTypes.reserve(mapping.size());
  for (auto indexAttr : mapping.getAsRange<IntegerAttr>()) {
    int64_t index = indexAttr.getInt();
    if (index < 0 || index >= static_cast<int64_t>(inputTypes.size()))
      return ::emitError(loc.value())
             << index << " is not a valid index into " << inputType;
    Type mappedType = inputTypes[index];
    outputTypes.push_back(mappedType);
  }

  // Create final tuple type.
  auto outputType = TupleType::get(context, outputTypes);
  inferredReturnTypes.push_back(outputType);

  return success();
}

/// Computes the type of the nested field of the given `type` identified by
/// `position`. Each entry `n` in the given index array `position` corresponds
/// to the `n`-th entry in that level. The function is thus implemented
/// recursively, where each recursion level extracts the type of the outer-most
/// level identified by the first index in the `position` array.
static FailureOr<Type> computeTypeAtPosition(Location loc, Type type,
                                             ArrayRef<int64_t> position) {
  if (position.empty())
    return type;

  // Recurse into tuple field of first index in position array.
  if (auto tupleType = llvm::dyn_cast<TupleType>(type)) {
    int64_t index = position[0];
    ArrayRef<Type> fieldTypes = tupleType.getTypes();
    if (index >= static_cast<int64_t>(fieldTypes.size()) || index < 0)
      return emitError(loc) << index << " is not a valid index for " << type;

    return computeTypeAtPosition(loc, fieldTypes[index], position.drop_front());
  }

  return emitError(loc) << "can't extract element from type " << type;
}

LogicalResult FieldReferenceOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
  auto *typedProperties = properties.as<Properties *>();
  if (!loc)
    loc = UnknownLoc::get(context);

  // Extract field type at given position.
  DenseI64ArrayAttr position = typedProperties->getPosition();
  Type inputType = operands[0].getType();
  FailureOr<Type> fieldType =
      computeTypeAtPosition(loc.value(), inputType, position);
  if (failed(fieldType))
    return ::emitError(loc.value())
           << "mismatching position and type (position: " << position
           << ", type: " << inputType << ")";

  inferredReturnTypes.push_back(fieldType.value());

  return success();
}

LogicalResult FilterOp::verifyRegions() {
  MLIRContext *context = getContext();
  Type si1 = IntegerType::get(context, /*width=*/1, IntegerType::Signed);
  Region &condition = getCondition();

  // Verify that type of yielded value is Boolean.
  auto yieldOp = llvm::cast<YieldOp>(condition.front().getTerminator());
  if (yieldOp.getValue().size() != 1)
    return emitOpError()
           << "must have 'condition' region yielding one value (yields "
           << yieldOp.getValue().size() << ")";

  Type yieldedType = yieldOp.getValue().getTypes()[0];
  if (yieldedType != si1)
    return emitOpError()
           << "must have 'condition' region yielding 'si1' (yields "
           << yieldedType << ")";

  // Verify that block has argument of input tuple type.
  Type tupleType = getResult().getType();
  if (condition.getNumArguments() != 1 ||
      condition.getArgument(0).getType() != tupleType) {
    InFlightDiagnostic diag = emitOpError()
                              << "must have 'condition' region taking "
                              << tupleType << " as argument (takes ";
    if (condition.getNumArguments() == 0)
      diag << "no arguments)";
    else
      diag << condition.getArgument(0).getType() << ")";
    return diag;
  }

  return success();
}

LogicalResult
LiteralOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc,
                            ValueRange operands, DictionaryAttr attributes,
                            OpaqueProperties properties, RegionRange regions,
                            llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
  auto *typedProperties = properties.as<Properties *>();
  Attribute valueAttr = typedProperties->getValue();

  Type resultType = getAttrType(valueAttr);
  if (!resultType)
    return emitOptionalError(loc, "unsuited attribute for literal value: ",
                             typedProperties->getValue());

  inferredReturnTypes.emplace_back(resultType);
  return success();
}

LogicalResult
JoinOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc,
                         ValueRange operands, DictionaryAttr attributes,
                         OpaqueProperties properties, RegionRange regions,
                         llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
  Value leftInput = operands[0];
  Value rightInput = operands[1];

  TypeRange leftFieldTypes = cast<TupleType>(leftInput.getType()).getTypes();
  TypeRange rightFieldTypes = cast<TupleType>(rightInput.getType()).getTypes();

  // Get accessor to `join_type`.
  Adaptor adaptor(operands, attributes, properties, regions);
  JoinTypeKind join_type = adaptor.getJoinType();

  SmallVector<mlir::Type> fieldTypes;

  switch (join_type) {
  case JoinTypeKind::unspecified:
  case JoinTypeKind::inner:
  case JoinTypeKind::outer:
  case JoinTypeKind::right:
  case JoinTypeKind::left:
    llvm::append_range(fieldTypes, leftFieldTypes);
    llvm::append_range(fieldTypes, rightFieldTypes);
    break;
  case JoinTypeKind::semi:
  case JoinTypeKind::anti:
    llvm::append_range(fieldTypes, leftFieldTypes);
    break;
  case JoinTypeKind::single:
    llvm::append_range(fieldTypes, rightFieldTypes);
    break;
  }

  auto resultType = TupleType::get(context, fieldTypes);

  inferredReturnTypes = SmallVector<Type>{resultType};

  return success();
}

/// Verifies that the provided field names match the provided field types. While
/// the field types are potentially nested, the names are given in a single,
/// flat list and correspond to the field types in depth first order (where each
/// nested tuple-typed field has a name and its nested field have names on their
/// own). Furthermore, the names on each nesting level need to be unique. For
/// details, see
/// https://substrait.io/tutorial/sql_to_substrait/#types-and-schemas.
static FailureOr<int>
verifyNamedStructHelper(Location loc, llvm::ArrayRef<Attribute> fieldNames,
                        TypeRange fieldTypes) {
  int numConsumedNames = 0;
  llvm::SmallSet<llvm::StringRef, 8> currentLevelNames;
  for (Type type : fieldTypes) {
    // Check name of current field.
    if (numConsumedNames >= static_cast<int>(fieldNames.size()))
      return emitError(loc, "not enough field names provided");
    auto currentName = llvm::cast<StringAttr>(fieldNames[numConsumedNames]);
    if (!currentLevelNames.insert(currentName).second)
      return emitError(loc, llvm::Twine("duplicate field name: '") +
                                currentName.getValue() + "'");
    numConsumedNames++;

    // Recurse for nested structs/tuples.
    if (auto tupleType = llvm::dyn_cast<TupleType>(type)) {
      llvm::ArrayRef<Type> nestedFieldTypes = tupleType.getTypes();
      llvm::ArrayRef<Attribute> remainingNames =
          fieldNames.drop_front(numConsumedNames);
      FailureOr<int> res =
          verifyNamedStructHelper(loc, remainingNames, nestedFieldTypes);
      if (failed(res))
        return failure();
      numConsumedNames += res.value();
    }
  }
  return numConsumedNames;
}

static LogicalResult verifyNamedStruct(Operation *op,
                                       llvm::ArrayRef<Attribute> fieldNames,
                                       TupleType tupleType) {
  Location loc = op->getLoc();
  TypeRange fieldTypes = tupleType.getTypes();

  // Emits error message with context on failure.
  auto emitErrorMessage = [&]() {
    InFlightDiagnostic error = op->emitOpError()
                               << "has mismatching 'field_names' ([";
    llvm::interleaveComma(fieldNames, error);
    error << "]) and result type (" << tupleType << ")";
    return error;
  };

  // Call recursive verification function.
  FailureOr<int> numConsumedNames =
      verifyNamedStructHelper(loc, fieldNames, fieldTypes);

  // Relay any failure.
  if (failed(numConsumedNames))
    return emitErrorMessage();

  // If we haven't consumed all names, we got too many of them, so report.
  if (numConsumedNames.value() != static_cast<int>(fieldNames.size())) {
    InFlightDiagnostic error = emitErrorMessage();
    error.attachNote(loc) << "too many field names provided";
    return error;
  }

  return success();
}

LogicalResult NamedTableOp::verify() {
  llvm::ArrayRef<Attribute> fieldNames = getFieldNames().getValue();
  auto tupleType = llvm::cast<TupleType>(getResult().getType());
  return verifyNamedStruct(getOperation(), fieldNames, tupleType);
}

LogicalResult PlanRelOp::verifyRegions() {
  // Verify that we `yield` exactly one value.
  auto yieldOp = llvm::cast<YieldOp>(getBody().front().getTerminator());
  if (yieldOp.getValue().size() != 1)
    return emitOpError()
           << "must have 'body' region yielding one value (yields "
           << yieldOp.getValue().size() << ")";

  // Verify that the field names match the field types. If we don't have any,
  // we're done.
  if (!getFieldNames().has_value())
    return success();

  // Otherwise, use helper to verify.
  llvm::ArrayRef<Attribute> fieldNames = getFieldNames()->getValue();
  auto tupleType = llvm::cast<TupleType>(yieldOp.getValue().getTypes()[0]);
  return verifyNamedStruct(getOperation(), fieldNames, tupleType);
}

OpFoldResult ProjectOp::fold(FoldAdaptor adaptor) {
  Operation *terminator = adaptor.getExpressions().front().getTerminator();

  // If the region does not yield any values, the the `project` has no effect.
  if (terminator->getNumOperands() == 0) {
    return getInput();
  }

  return {};
}

LogicalResult ProjectOp::verifyRegions() {
  // Verify that the expression block has a matching argument type.
  auto inputTupleType = llvm::cast<TupleType>(getInput().getType());
  auto blockArgTypes = getExpressions().front().getArgumentTypes();
  if (blockArgTypes != ArrayRef<Type>(inputTupleType))
    return emitOpError()
           << "has 'expressions' region with mismatching argument type"
           << " (has: " << blockArgTypes << ", expected: " << inputTupleType
           << ")";

  // Verify that the input field types are a prefix of the output field types.
  size_t numInputFields = inputTupleType.getTypes().size();
  auto outputTupleType = llvm::cast<TupleType>(getResult().getType());
  ArrayRef<Type> outputPrefixTypes =
      outputTupleType.getTypes().take_front(numInputFields);

  if (inputTupleType.getTypes() != outputPrefixTypes)
    return emitOpError()
           << "has output field type whose prefix is different from "
           << "input field types (" << inputTupleType.getTypes() << " vs "
           << outputPrefixTypes << ")";

  // Verify that yielded operands have the same types as the new output fields.
  ArrayRef<Type> newFieldTypes =
      outputTupleType.getTypes().drop_front(numInputFields);
  auto yieldOp = llvm::cast<YieldOp>(getExpressions().front().getTerminator());

  if (yieldOp.getOperandTypes() != newFieldTypes)
    return emitOpError()
           << "has output field type whose new fields are different from "
           << "the yielded operand types (" << newFieldTypes << " vs "
           << yieldOp.getOperandTypes() << ")";

  return success();
}

} // namespace substrait
} // namespace mlir

//===----------------------------------------------------------------------===//
// Substrait types and attributes
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "substrait-mlir/Dialect/Substrait/IR/SubstraitOpsTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "substrait-mlir/Dialect/Substrait/IR/SubstraitOpsAttrs.cpp.inc"
