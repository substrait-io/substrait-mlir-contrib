//===-- Substrait.cpp - Substrait dialect -----------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "substrait-mlir/Dialect/Substrait/IR/Substrait.h"

#include "mlir/IR/DialectImplementation.h" // IWYU pragma: keep
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h" // IWYU pragma: keep
#include "llvm/Support/Regex.h"

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

LogicalResult mlir::substrait::IntervalYearMonthAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, int32_t year,
    int32_t month) {
  if (year < -100000 || year > 100000)
    return emitError() << "year must be in a range of [-10,000..10,000] years";
  if (month < -120000 || month > 120000)
    return emitError()
           << "month must be in a range of [120,000..120,000] months";
  return success();
}

LogicalResult mlir::substrait::IntervalDaySecondAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, int32_t days,
    int32_t seconds) {
  if (days < -3650000 || days > 3650000)
    return emitError()
           << "days must be in a range of [-3,650,000..3,650,000] days";
  // The value of `seconds` should be within the range [-315,360,000,000..
  // 315,360,000,000]. However, this exceeds the limits of int32_t (as required
  // by the specification), making it untestable within the given constraints.
  return success();
}

//===----------------------------------------------------------------------===//
// Substrait types
//===----------------------------------------------------------------------===//

LogicalResult mlir::substrait::DecimalType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    uint32_t precision, uint32_t scale) {
  if (precision > 38)
    return emitError() << "precision must be in a range of [0..38] but got "
                       << precision;

  if (scale > precision)
    return emitError() << "scale must be in a range of [0..P] (P = "
                       << precision << ") but got " << scale;

  return success();
}

// Count the number of digits in an APInt in base 10.
static size_t countDigits(const APInt &value) {
  llvm::SmallVector<char> buffer;
  value.toString(buffer, 10, /*isSigned=*/false);
  return buffer.size();
}

LogicalResult mlir::substrait::DecimalAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, DecimalType type,
    IntegerAttr value) {

  if (value.getType().getIntOrFloatBitWidth() != 128)
    return emitError() << "value must be a 128-bit integer";

  // Max 38 digits.
  size_t nDigits = countDigits(value.getValue());
  size_t P = type.getPrecision();
  if (nDigits > P)
    return emitError() << "value must have at most '" << P
                       << "' digits as per the type '" << type << "' but got "
                       << nDigits;

  return success();
}

std::string DecimalAttr::decimalStr() const {
  SmallVector<char> buffer;
  getValue().getValue().toString(buffer, 10, /*isSigned=*/false);

  // Pad buffer up to P digits with leading zeros.
  buffer.insert(buffer.begin(), getType().getPrecision() - buffer.size(), '0');

  size_t scale = getType().getScale();
  assert(scale <= buffer.size() && "scale must be <= precision");

  // Insert the decimal point.
  buffer.insert(buffer.end() - scale, '.');

  // Trim trailing and leading zeros
  auto ref = StringRef(buffer.data(), buffer.size());
  size_t firstNonZero = ref.find_first_not_of('0');
  if (firstNonZero != StringRef::npos)
    ref = ref.drop_front(firstNonZero);

  size_t lastNonZero = ref.find_last_not_of('0');
  if (lastNonZero != StringRef::npos)
    ref = ref.substr(0, lastNonZero + 1);

  std::string res = ref.str();

  // Edge cases: no integer and no fractional parts. In these cases, we want to
  // have a single trailing/leading zero.
  if (res.front() == '.')
    res = "0" + res;
  if (res.back() == '.')
    res = res + "0";

  return res;
}

Attribute DecimalAttr::parse(AsmParser &odsParser, Type odsType) {
  Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  DecimalType type;
  if (odsParser.parseLess())
    return {};

  std::string value;
  if (odsParser.parseString(&value) || odsParser.parseColon() ||
      odsParser.parseType(type))
    return {};

  if (odsParser.parseGreater())
    return {};

  return odsParser.getChecked<DecimalAttr>(odsLoc, odsParser.getContext(), type,
                                           mlir::StringRef(value));
}

void DecimalAttr::print(AsmPrinter &odsPrinter) const {
  Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << "\"" << decimalStr() << "\"";
  odsPrinter << " : ";
  odsPrinter.printType(getType());
  odsPrinter << ">";
}

DecimalAttr DecimalAttr::get(::mlir::MLIRContext *context, DecimalType type,
                             StringRef value) {
  return getChecked(nullptr, context, type, value);
}
DecimalAttr DecimalAttr::getChecked(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::mlir::MLIRContext *context, DecimalType type, StringRef value) {
  // Sanity check: contains only digits and a single decimal point.
  static constexpr std::string_view regex = "^[0-9]+\\.[0-9]+$";
  auto r = llvm::Regex(regex);
  if (!r.match(value)) {
    emitError() << "invalid decimal value: " << value;
    return nullptr;
  }

  // Trim trailing zeros.
  size_t lastNonZero = value.find_last_not_of('0');
  if (lastNonZero == StringRef::npos)
    lastNonZero = value.size() - 1;
  value = value.substr(0, lastNonZero + 1);

  // Verify scale.
  size_t decimalPos = value.find('.');
  size_t detectedScale = value.size() - decimalPos - 1;
  if (detectedScale > type.getScale()) {
    emitError()
        << "decimal value has incorrect number of digits after the decimal "
        << "point (" << detectedScale << "). Expected <=" << type.getScale()
        << " as per the type " << type;
    return nullptr;
  }

  // Add trailing zeros if necessary (detectedScale != type.getScale()). This is
  // required to be able to represent values where the number of digits after
  // the decimal point is less than the scale.
  std::string baseValueStr = value.str();
  if (detectedScale < type.getScale()) {
    baseValueStr.append(type.getScale() - detectedScale, '0');
  }

  // Parse the value by removing the decimal point.
  baseValueStr.erase(std::remove(baseValueStr.begin(), baseValueStr.end(), '.'),
                     baseValueStr.end());

  size_t nDigits = baseValueStr.size();

  if (nDigits > 38) {
    emitError() << "decimal value has too many digits (" << nDigits
                << "). Expected at most 38 digits";
    return nullptr;
  }

  APInt intValue(128, baseValueStr, 10);
  auto iType = IntegerType::get(context, 128);
  auto iAttr = IntegerAttr::getChecked(emitError, iType, intValue);
  return DecimalAttr::getChecked(emitError, context, type, iAttr);
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

namespace mlir {
namespace substrait {

static ParseResult
parseAggregationInvocation(OpAsmParser &parser,
                           AggregationInvocationAttr &aggregationInvocation);
static void
printAggregationInvocation(OpAsmPrinter &printer, CallOp op,
                           AggregationInvocationAttr aggregationInvocation);
static ParseResult parseAggregateRegions(OpAsmParser &parser,
                                         Region &groupingsRegion,
                                         Region &measuresRegion,
                                         ArrayAttr &groupingSetsAttr);
static void printAggregateRegions(OpAsmPrinter &printer, AggregateOp op,
                                  Region &groupingsRegion,
                                  Region &measuresRegion,
                                  ArrayAttr groupingSetsAttr);

} // namespace substrait
} // namespace mlir

#define GET_OP_CLASSES
#include "substrait-mlir/Dialect/Substrait/IR/SubstraitOps.cpp.inc"

namespace {

/// Computes the type of the nested field of the given `type` identified by
/// `position`. Each entry `n` in the given index array `position` corresponds
/// to the `n`-th entry in that level. The function is thus implemented
/// recursively, where each recursion level extracts the type of the outer-most
/// level identified by the first index in the `position` array.
FailureOr<Type> computeTypeAtPosition(Location loc, Type type,
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

/// Verifies that the provided field names match the provided field types. While
/// the field types are potentially nested, the names are given in a single,
/// flat list and correspond to the field types in depth first order (where each
/// nested tuple-typed field has a name and its nested field have names on their
/// own). Furthermore, the names on each nesting level need to be unique. For
/// details, see
/// https://substrait.io/tutorial/sql_to_substrait/#types-and-schemas.
FailureOr<int> verifyNamedStructHelper(Location loc,
                                       llvm::ArrayRef<Attribute> fieldNames,
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

LogicalResult verifyNamedStruct(Operation *op,
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

} // namespace

namespace mlir {
namespace substrait {

ParseResult
parseAggregationInvocation(OpAsmParser &parser,
                           AggregationInvocationAttr &aggregationInvocation) {
  // This is essentially copied from `FieldParser<AggregationInvocation>` but
  // sets the default `all` case if no invocation type is present.

  MLIRContext *context = parser.getContext();
  std::string keyword;
  if (failed(parser.parseOptionalKeywordOrString(&keyword))) {
    // No keyword parse --> use default value.
    aggregationInvocation =
        AggregationInvocationAttr::get(context, AggregationInvocation::all);
    return success();
  }

  // Symbolize the keyword.
  if (std::optional<AggregationInvocation> attr =
          symbolizeAggregationInvocation(keyword)) {
    aggregationInvocation =
        AggregationInvocationAttr::get(parser.getContext(), attr.value());
    return success();
  }

  // Symbolization failed.
  auto loc = parser.getCurrentLocation();
  return parser.emitError(loc)
         << "has invalid aggregate invocation type specification: " << keyword;
}

void printAggregationInvocation(
    OpAsmPrinter &printer, CallOp op,
    AggregationInvocationAttr aggregationInvocation) {
  if (aggregationInvocation &&
      aggregationInvocation.getValue() != AggregationInvocation::all) {
    // The whitespace printed here compensates the trimming of whitespace in
    // the declarative assembly format.
    printer << " " << aggregationInvocation.getValue();
  }
}

ParseResult parseAggregateRegions(OpAsmParser &parser, Region &groupingsRegion,
                                  Region &measuresRegion,
                                  ArrayAttr &groupingSetsAttr) {
  MLIRContext *context = parser.getContext();

  // Parse `measures` and `groupings` regions as well as `grouping_sets` attr.
  bool hasMeasures = false;
  bool hasGroupings = false;
  bool hasGroupingSets = false;
  {
    auto ensureOneOccurrance = [&](bool &hasParsed,
                                   StringRef name) -> LogicalResult {
      if (hasParsed) {
        SMLoc loc = parser.getCurrentLocation();
        return parser.emitError(loc, llvm::Twine("can only have one ") + name);
      }
      hasParsed = true;
      return success();
    };

    StringRef keyword;
    while (succeeded(parser.parseOptionalKeyword(
        &keyword, {"measures", "groupings", "grouping_sets"}))) {
      if (keyword == "measures") {
        if (failed(ensureOneOccurrance(hasMeasures, "'measures' region")) ||
            failed(parser.parseRegion(measuresRegion)))
          return failure();
      } else if (keyword == "groupings") {
        if (failed(ensureOneOccurrance(hasGroupings, "'groupings' region")) ||
            failed(parser.parseRegion(groupingsRegion)))
          return failure();
      } else if (keyword == "grouping_sets") {
        if (failed(ensureOneOccurrance(hasGroupingSets,
                                       "'grouping_sets' attribute")) ||
            failed(parser.parseAttribute(groupingSetsAttr)))
          return failure();
      }
    }
  }

  // Create default value of `grouping_sets` attr if not provided.
  if (!hasGroupingSets) {
    // If there is no `groupings` region, create only the empty grouping set.
    if (!hasGroupings)
      groupingSetsAttr = ArrayAttr::get(context, ArrayAttr::get(context, {}));
    // Otherwise, create the grouping set with all grouping columns.
    else if (!groupingsRegion.empty()) {
      auto yieldOp =
          llvm::dyn_cast<YieldOp>(groupingsRegion.front().getTerminator());
      if (yieldOp) {
        unsigned numColumns = yieldOp->getNumOperands();
        SmallVector<int64_t> allColumns;
        llvm::append_range(allColumns, llvm::seq(0u, numColumns));
        IRRewriter rewriter(context);
        ArrayAttr allColumnsAttr = rewriter.getI64ArrayAttr(allColumns);
        groupingSetsAttr = rewriter.getArrayAttr({allColumnsAttr});
      }
    }
  }

  return success();
}

void printAggregateRegions(OpAsmPrinter &printer, AggregateOp op,
                           Region &groupingsRegion, Region &measuresRegion,
                           ArrayAttr groupingSetsAttr) {
  printer.increaseIndent();

  // `groupings` region.
  if (!groupingsRegion.empty()) {
    printer.printNewline();
    printer.printKeywordOrString("groupings");
    printer << " ";
    printer.printRegion(groupingsRegion);
  }

  // `grouping_sets` attribute.
  if (groupingSetsAttr.size() != 1) {
    // Note: A single grouping set is always of the form `seq(0, size)`.
    printer.printNewline();
    printer.printKeywordOrString("grouping_sets");
    printer << " ";
    printer.printAttribute(groupingSetsAttr);
  }

  // `measures` regions.
  if (!measuresRegion.empty()) {
    printer.printNewline();
    printer.printKeywordOrString("measures");
    printer << " ";
    printer.printRegion(measuresRegion);
  }

  printer.decreaseIndent();
}

void AggregateOp::build(OpBuilder &builder, OperationState &result, Value input,
                        ArrayAttr groupingSets, Region *groupings,
                        Region *measures) {

  MLIRContext *context = builder.getContext();
  auto loc = UnknownLoc::get(context);
  AggregateOp::Properties properties;
  properties.grouping_sets = groupingSets;
  SmallVector<Region *> regions = {groupings, measures};

  // Infer `returnTypes` from provided arguments. If that fails, then
  // `returnType` will be empty. The rest of this function will continue to
  // work, but the op that is built in the end will not verify and the
  // diagnostics of `inferReturnType` will have been emitted.
  SmallVector<mlir::Type> returnTypes;
  (void)AggregateOp::inferReturnTypes(context, loc, input, {},
                                      OpaqueProperties(&properties), regions,
                                      returnTypes);

  // Call existing `build` function and move bodies into the new regions.
  AggregateOp::build(builder, result, returnTypes, input, groupingSets);
  result.regions[0]->takeBody(*groupings);
  result.regions[1]->takeBody(*measures);
}

LogicalResult AggregateOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {
  auto *typedProperties = properties.as<Properties *>();
  assert(typedProperties && "could not get typed properties");
  Region *groupings = regions[0];
  Region *measures = regions[1];
  SmallVector<Type> fieldTypes;
  if (!loc)
    loc = UnknownLoc::get(context);

  // The left-most output columns are the `groupings` columns, then the
  // `measures` columns.
  for (Region *region : {groupings, measures}) {
    if (region->empty())
      continue;
    auto yieldOp = llvm::cast<YieldOp>(region->front().getTerminator());
    llvm::append_range(fieldTypes, yieldOp.getOperandTypes());
  }

  // If there is more than one `grouping_set`, then we also have an additional
  // `si32` column for the grouping set ID.
  if (typedProperties->grouping_sets.size() > 1) {
    auto si32 = IntegerType::get(context, /*width=*/32, IntegerType::Signed);
    fieldTypes.push_back(si32);
  }

  // Build tuple type from field types.
  auto resultType = TupleType::get(context, fieldTypes);
  inferredReturnTypes.push_back(resultType);

  return success();
}

LogicalResult AggregateOp::verifyRegions() {
  // Verify properties that need to hold for both regions.
  auto inputTupleType = getInput().getType();
  for (auto [idx, region] : llvm::enumerate(getRegions())) {
    if (region->empty()) // Regions are allowed to be empty.
      continue;

    // Verify that the regions have the input tuple as argument.
    if (region->getArgumentTypes() != TypeRange{inputTupleType})
      return emitOpError() << "has region #" << idx
                           << " with invalid argument types (expected: "
                           << inputTupleType
                           << ", got: " << region->getArgumentTypes() << ")";

    // Verify that at least one value is yielded.
    auto yieldOp = llvm::cast<YieldOp>(region->front().getTerminator());
    if (yieldOp->getNumOperands() == 0)
      return emitOpError()
             << "has region #" << idx
             << " that yields no values (use empty region instead)";
  }

  // Verify that the grouping sets refer to values yielded from `groupings`,
  // that all yielded values are referred to, and that the references are in the
  // correct order.
  {
    int64_t numGroupingColumns = 0;
    if (!getGroupings().empty()) {
      auto yieldOp =
          llvm::cast<YieldOp>(getGroupings().front().getTerminator());
      numGroupingColumns = yieldOp->getNumOperands();
    }

    // Check bounds, collect grouping columns.
    llvm::SmallSet<int64_t, 16> allGroupingRefs;
    for (auto [groupingSetIdx, groupingSet] :
         llvm::enumerate(getGroupingSets())) {
      for (auto [refIdx, refAttr] :
           llvm::enumerate(cast<ArrayAttr>(groupingSet))) {
        auto ref = cast<IntegerAttr>(refAttr).getInt();
        if (ref < 0 || ref >= numGroupingColumns)
          return emitOpError() << "has invalid grouping set #" << groupingSetIdx
                               << ": column reference " << ref << " (column #"
                               << refIdx << ") is out of bounds";
        auto [_, hasInserted] = allGroupingRefs.insert(ref);
        if (hasInserted &&
            ref != static_cast<int64_t>(allGroupingRefs.size() - 1))
          return emitOpError()
                 << "has invalid grouping sets: the first occerrences of the "
                    "column references must be densely increasing";
      }
    }

    // Check that all grouping columns are used.
    if (static_cast<int64_t>(allGroupingRefs.size()) != numGroupingColumns) {
      for (int64_t i : llvm::seq<int64_t>(0, numGroupingColumns)) {
        if (!allGroupingRefs.contains(i))
          return emitOpError() << "has 'groupings' region whose operand #" << i
                               << " is not contained in any 'grouping_set'";
      }
    }
  }

  // Verify that `measures` region yields only values produced by
  // `AggregateFunction`s.
  if (!getMeasures().empty()) {
    for (Value value : getMeasures().front().getTerminator()->getOperands()) {
      auto callOp = llvm::dyn_cast_or_null<CallOp>(value.getDefiningOp());
      if (!callOp || !callOp.isAggregate())
        return emitOpError() << "yields value from 'measures' region that was "
                                "not produced by an aggregate function: "
                             << value;
    }
  }

  if (getGroupings().empty() && getMeasures().empty())
    return emitOpError()
           << "one of 'groupings' or 'measures' must be specified";

  return success();
}

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

LogicalResult ExtensionTableOp::verify() {
  llvm::ArrayRef<Attribute> fieldNames = getFieldNames().getValue();
  auto tupleType = llvm::cast<TupleType>(getResult().getType());
  return verifyNamedStruct(getOperation(), fieldNames, tupleType);
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
  JoinType join_type = adaptor.getJoinType();

  SmallVector<mlir::Type> fieldTypes;

  switch (join_type) {
  case JoinType::unspecified:
  case JoinType::inner:
  case JoinType::outer:
  case JoinType::right:
  case JoinType::left:
    llvm::append_range(fieldTypes, leftFieldTypes);
    llvm::append_range(fieldTypes, rightFieldTypes);
    break;
  case JoinType::semi:
  case JoinType::anti:
    llvm::append_range(fieldTypes, leftFieldTypes);
    break;
  case JoinType::single:
    llvm::append_range(fieldTypes, rightFieldTypes);
    break;
  }

  auto resultType = TupleType::get(context, fieldTypes);

  inferredReturnTypes = SmallVector<Type>{resultType};

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
