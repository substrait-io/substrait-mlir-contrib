# Coverage Stats

This file shows how much of the Substrait specification this project currently
covers. The coverage is computed by comparing the number of fields in the
specification (i.e., the `.proto` files) with the number of fields occurring
in the unit tests in this repository. This is approximate in several ways, for
example, does it not account for how many enum values are covered, but it is
reasonably easy to compute and should still provide an indication for progress.

## Coverage per `.proto` file

The following table shows the coverage per `.proto` file, i.e., for each of the
`.proto` files in the main Substrait repository, it counts all fields at all
nesting level as the `total` and compares that with the number of fields that
occur in the unit tests.

| file                              |   covered (number) |   total | covered (fraction)   |
|-----------------------------------|--------------------|---------|----------------------|
| proto/extensions/extensions.proto |                 16 |      16 | 100 %                |
| proto/plan.proto                  |                 13 |      14 | 92 %                 |
| proto/type.proto                  |                 50 |      92 | 54 %                 |
| proto/algebra.proto               |                102 |     381 | 26 %                 |
| proto/capabilities.proto          |                  0 |       7 | 0 %                  |
| proto/extended_expression.proto   |                  0 |      10 | 0 %                  |
| proto/function.proto              |                  0 |      50 | 0 %                  |
| proto/parameterized_types.proto   |                  0 |      62 | 0 %                  |
| proto/type_expressions.proto      |                  0 |      72 | 0 %                  |

## Coverage per top-level message type

The following table shows a drill-down into the top-level message types of each
file, i.e., for each file, for each message type defined at the top level of
that file, the number of `total` fields is counted and compare to the number of
fields that are covered in the unit tests.

| file                              | top-level message type       |   covered (number) |   total | covered (fraction)   |
|-----------------------------------|------------------------------|--------------------|---------|----------------------|
| proto/algebra.proto               | CrossRel                     |                  4 |       4 | 100 %                |
| proto/algebra.proto               | FetchRel                     |                  5 |       5 | 100 %                |
| proto/algebra.proto               | FilterRel                    |                  4 |       4 | 100 %                |
| proto/algebra.proto               | ProjectRel                   |                  4 |       4 | 100 %                |
| proto/algebra.proto               | RelRoot                      |                  2 |       2 | 100 %                |
| proto/algebra.proto               | SetRel                       |                  4 |       4 | 100 %                |
| proto/extensions/extensions.proto | AdvancedExtension            |                  2 |       2 | 100 %                |
| proto/extensions/extensions.proto | SimpleExtensionDeclaration   |                 12 |      12 | 100 %                |
| proto/extensions/extensions.proto | SimpleExtensionURI           |                  2 |       2 | 100 %                |
| proto/plan.proto                  | Plan                         |                  6 |       6 | 100 %                |
| proto/plan.proto                  | PlanRel                      |                  2 |       2 | 100 %                |
| proto/plan.proto                  | Version                      |                  5 |       5 | 100 %                |
| proto/type.proto                  | NamedStruct                  |                  2 |       2 | 100 %                |
| proto/algebra.proto               | AggregateRel                 |                  7 |       8 | 87 %                 |
| proto/algebra.proto               | JoinRel                      |                  5 |       7 | 71 %                 |
| proto/type.proto                  | Type                         |                 48 |      90 | 53 %                 |
| proto/algebra.proto               | AggregateFunction            |                  4 |       8 | 50 %                 |
| proto/algebra.proto               | Rel                          |                  8 |      21 | 38 %                 |
| proto/algebra.proto               | FunctionArgument             |                  1 |       3 | 33 %                 |
| proto/algebra.proto               | Expression                   |                 44 |     167 | 26 %                 |
| proto/algebra.proto               | ReadRel                      |                  7 |      28 | 25 %                 |
| proto/algebra.proto               | RelCommon                    |                  3 |      12 | 25 %                 |
| proto/algebra.proto               | ComparisonJoinKey            |                  0 |       5 | 0 %                  |
| proto/algebra.proto               | ConsistentPartitionWindowRel |                  0 |      15 | 0 %                  |
| proto/algebra.proto               | DdlRel                       |                  0 |       8 | 0 %                  |
| proto/algebra.proto               | ExchangeRel                  |                  0 |      18 | 0 %                  |
| proto/algebra.proto               | ExpandRel                    |                  0 |       6 | 0 %                  |
| proto/algebra.proto               | ExtensionLeafRel             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | ExtensionMultiRel            |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | ExtensionObject              |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | ExtensionSingleRel           |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | FunctionOption               |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | HashJoinRel                  |                  0 |       9 | 0 %                  |
| proto/algebra.proto               | MergeJoinRel                 |                  0 |       9 | 0 %                  |
| proto/algebra.proto               | NamedObjectWrite             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | NestedLoopJoinRel            |                  0 |       6 | 0 %                  |
| proto/algebra.proto               | ReferenceRel                 |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | SortField                    |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | SortRel                      |                  0 |       4 | 0 %                  |
| proto/algebra.proto               | WriteRel                     |                  0 |       7 | 0 %                  |
| proto/capabilities.proto          | Capabilities                 |                  0 |       7 | 0 %                  |
| proto/extended_expression.proto   | ExpressionReference          |                  0 |       3 | 0 %                  |
| proto/extended_expression.proto   | ExtendedExpression           |                  0 |       7 | 0 %                  |
| proto/function.proto              | FunctionSignature            |                  0 |      50 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            |                  0 |      62 | 0 %                  |
| proto/plan.proto                  | PlanVersion                  |                  0 |       1 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         |                  0 |      72 | 0 %                  |

## Coverage per `Type` and `Expression` submessage types

The following table shows a drill-down into the `Type` and `Expression`
submessages. Those are some of the most interesting parts of the specification
but not defined in top-level message types, so they are not visible in the
previous table.

| file                | msg_lvl0   | msg_lvl1         |   covered (number) |   total | covered (fraction)   |
|---------------------|------------|------------------|--------------------|---------|----------------------|
| proto/algebra.proto | Expression | Cast             |                  3 |       3 | 100 %                |
| proto/type.proto    | Type       | Decimal          |                  3 |       4 | 75 %                 |
| proto/type.proto    | Type       | FixedBinary      |                  2 |       3 | 66 %                 |
| proto/type.proto    | Type       | FixedChar        |                  2 |       3 | 66 %                 |
| proto/type.proto    | Type       | Struct           |                  2 |       3 | 66 %                 |
| proto/type.proto    | Type       | VarChar          |                  2 |       3 | 66 %                 |
| proto/algebra.proto | Expression | ScalarFunction   |                  3 |       5 | 60 %                 |
| proto/algebra.proto | Expression | Literal          |                 28 |      47 | 59 %                 |
| proto/algebra.proto | Expression | FieldReference   |                  3 |       6 | 50 %                 |
| proto/type.proto    | Type       | Binary           |                  1 |       2 | 50 %                 |
| proto/type.proto    | Type       | Boolean          |                  1 |       2 | 50 %                 |
| proto/type.proto    | Type       | Date             |                  1 |       2 | 50 %                 |
| proto/type.proto    | Type       | FP32             |                  1 |       2 | 50 %                 |
| proto/type.proto    | Type       | FP64             |                  1 |       2 | 50 %                 |
| proto/type.proto    | Type       | I16              |                  1 |       2 | 50 %                 |
| proto/type.proto    | Type       | I32              |                  1 |       2 | 50 %                 |
| proto/type.proto    | Type       | I64              |                  1 |       2 | 50 %                 |
| proto/type.proto    | Type       | I8               |                  1 |       2 | 50 %                 |
| proto/type.proto    | Type       | IntervalDay      |                  1 |       2 | 50 %                 |
| proto/type.proto    | Type       | IntervalYear     |                  1 |       2 | 50 %                 |
| proto/type.proto    | Type       | String           |                  1 |       2 | 50 %                 |
| proto/type.proto    | Type       | Time             |                  1 |       2 | 50 %                 |
| proto/type.proto    | Type       | Timestamp        |                  1 |       2 | 50 %                 |
| proto/type.proto    | Type       | TimestampTZ      |                  1 |       2 | 50 %                 |
| proto/type.proto    | Type       | UUID             |                  1 |       2 | 50 %                 |
| proto/algebra.proto | Expression | ReferenceSegment |                  3 |       9 | 33 %                 |
| proto/algebra.proto | Expression | EmbeddedFunction |                  0 |       8 | 0 %                  |
| proto/algebra.proto | Expression | Enum             |                  0 |       2 | 0 %                  |
| proto/algebra.proto | Expression | IfThen           |                  0 |       4 | 0 %                  |
| proto/algebra.proto | Expression | MaskExpression   |                  0 |      20 | 0 %                  |
| proto/algebra.proto | Expression | MultiOrList      |                  0 |       3 | 0 %                  |
| proto/algebra.proto | Expression | Nested           |                  0 |      10 | 0 %                  |
| proto/algebra.proto | Expression | SingularOrList   |                  0 |       2 | 0 %                  |
| proto/algebra.proto | Expression | Subquery         |                  0 |      13 | 0 %                  |
| proto/algebra.proto | Expression | SwitchExpression |                  0 |       5 | 0 %                  |
| proto/algebra.proto | Expression | WindowFunction   |                  0 |      18 | 0 %                  |
| proto/type.proto    | Type       | List             |                  0 |       3 | 0 %                  |
| proto/type.proto    | Type       | Map              |                  0 |       4 | 0 %                  |
| proto/type.proto    | Type       | Parameter        |                  0 |       6 | 0 %                  |
| proto/type.proto    | Type       | UserDefined      |                  0 |       4 | 0 %                  |
