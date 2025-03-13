| file                              |   covered (number) |   total | covered (fraction)   |
|-----------------------------------|--------------------|---------|----------------------|
| proto/extensions/extensions.proto |                 16 |      16 | 100 %                |
| proto/plan.proto                  |                 13 |      14 | 92 %                 |
| proto/type.proto                  |                 44 |      92 | 47 %                 |
| proto/algebra.proto               |                 92 |     381 | 24 %                 |
| proto/capabilities.proto          |                  0 |       7 | 0 %                  |
| proto/extended_expression.proto   |                  0 |      10 | 0 %                  |
| proto/function.proto              |                  0 |      50 | 0 %                  |
| proto/parameterized_types.proto   |                  0 |      62 | 0 %                  |
| proto/type_expressions.proto      |                  0 |      72 | 0 %                  |

| file                              | msg_lvl0                     |   covered (number) |   total | covered (fraction)   |
|-----------------------------------|------------------------------|--------------------|---------|----------------------|
| proto/algebra.proto               | ProjectRel                   |                  4 |       4 | 100 %                |
| proto/algebra.proto               | RelRoot                      |                  2 |       2 | 100 %                |
| proto/extensions/extensions.proto | AdvancedExtension            |                  2 |       2 | 100 %                |
| proto/extensions/extensions.proto | SimpleExtensionDeclaration   |                 12 |      12 | 100 %                |
| proto/extensions/extensions.proto | SimpleExtensionURI           |                  2 |       2 | 100 %                |
| proto/plan.proto                  | Plan                         |                  6 |       6 | 100 %                |
| proto/plan.proto                  | PlanRel                      |                  2 |       2 | 100 %                |
| proto/plan.proto                  | Version                      |                  5 |       5 | 100 %                |
| proto/type.proto                  | NamedStruct                  |                  2 |       2 | 100 %                |
| proto/algebra.proto               | FetchRel                     |                  4 |       5 | 80 %                 |
| proto/algebra.proto               | AggregateRel                 |                  6 |       8 | 75 %                 |
| proto/algebra.proto               | CrossRel                     |                  3 |       4 | 75 %                 |
| proto/algebra.proto               | FilterRel                    |                  3 |       4 | 75 %                 |
| proto/algebra.proto               | SetRel                       |                  3 |       4 | 75 %                 |
| proto/algebra.proto               | JoinRel                      |                  4 |       7 | 57 %                 |
| proto/algebra.proto               | AggregateFunction            |                  4 |       8 | 50 %                 |
| proto/type.proto                  | Type                         |                 42 |      90 | 46 %                 |
| proto/algebra.proto               | Rel                          |                  8 |      21 | 38 %                 |
| proto/algebra.proto               | FunctionArgument             |                  1 |       3 | 33 %                 |
| proto/algebra.proto               | RelCommon                    |                  3 |      12 | 25 %                 |
| proto/algebra.proto               | Expression                   |                 41 |     167 | 24 %                 |
| proto/algebra.proto               | ReadRel                      |                  6 |      28 | 21 %                 |
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

| file                              | msg_lvl0                     | msg_lvl1                 |   covered (number) |   total | covered (fraction)   |
|-----------------------------------|------------------------------|--------------------------|--------------------|---------|----------------------|
| proto/algebra.proto               | AggregateRel                 | Grouping                 |                  1 |       1 | 100 %                |
| proto/algebra.proto               | Expression                   | Cast                     |                  3 |       3 | 100 %                |
| proto/algebra.proto               | ProjectRel                   |                          |                  4 |       4 | 100 %                |
| proto/algebra.proto               | ReadRel                      | ExtensionTable           |                  1 |       1 | 100 %                |
| proto/algebra.proto               | RelCommon                    | Emit                     |                  1 |       1 | 100 %                |
| proto/algebra.proto               | RelRoot                      |                          |                  2 |       2 | 100 %                |
| proto/extensions/extensions.proto | AdvancedExtension            |                          |                  2 |       2 | 100 %                |
| proto/extensions/extensions.proto | SimpleExtensionDeclaration   |                          |                  3 |       3 | 100 %                |
| proto/extensions/extensions.proto | SimpleExtensionDeclaration   | ExtensionFunction        |                  3 |       3 | 100 %                |
| proto/extensions/extensions.proto | SimpleExtensionDeclaration   | ExtensionType            |                  3 |       3 | 100 %                |
| proto/extensions/extensions.proto | SimpleExtensionDeclaration   | ExtensionTypeVariation   |                  3 |       3 | 100 %                |
| proto/extensions/extensions.proto | SimpleExtensionURI           |                          |                  2 |       2 | 100 %                |
| proto/plan.proto                  | Plan                         |                          |                  6 |       6 | 100 %                |
| proto/plan.proto                  | PlanRel                      |                          |                  2 |       2 | 100 %                |
| proto/plan.proto                  | Version                      |                          |                  5 |       5 | 100 %                |
| proto/type.proto                  | NamedStruct                  |                          |                  2 |       2 | 100 %                |
| proto/algebra.proto               | AggregateRel                 |                          |                  4 |       5 | 80 %                 |
| proto/algebra.proto               | FetchRel                     |                          |                  4 |       5 | 80 %                 |
| proto/type.proto                  | Type                         |                          |                 19 |      25 | 76 %                 |
| proto/algebra.proto               | CrossRel                     |                          |                  3 |       4 | 75 %                 |
| proto/algebra.proto               | FilterRel                    |                          |                  3 |       4 | 75 %                 |
| proto/algebra.proto               | SetRel                       |                          |                  3 |       4 | 75 %                 |
| proto/type.proto                  | Type                         | Decimal                  |                  3 |       4 | 75 %                 |
| proto/type.proto                  | Type                         | FixedChar                |                  2 |       3 | 66 %                 |
| proto/type.proto                  | Type                         | Struct                   |                  2 |       3 | 66 %                 |
| proto/algebra.proto               | Expression                   | ScalarFunction           |                  3 |       5 | 60 %                 |
| proto/algebra.proto               | JoinRel                      |                          |                  4 |       7 | 57 %                 |
| proto/algebra.proto               | Expression                   | Literal                  |                 25 |      47 | 53 %                 |
| proto/algebra.proto               | AggregateFunction            |                          |                  4 |       8 | 50 %                 |
| proto/algebra.proto               | AggregateRel                 | Measure                  |                  1 |       2 | 50 %                 |
| proto/algebra.proto               | Expression                   | FieldReference           |                  3 |       6 | 50 %                 |
| proto/algebra.proto               | ReadRel                      | NamedTable               |                  1 |       2 | 50 %                 |
| proto/algebra.proto               | RelCommon                    |                          |                  2 |       4 | 50 %                 |
| proto/type.proto                  | Type                         | Binary                   |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | Boolean                  |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | Date                     |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | FP32                     |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | FP64                     |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | I16                      |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | I32                      |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | I64                      |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | I8                       |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | IntervalDay              |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | IntervalYear             |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | String                   |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | Time                     |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | Timestamp                |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | TimestampTZ              |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | UUID                     |                  1 |       2 | 50 %                 |
| proto/algebra.proto               | ReadRel                      |                          |                  4 |      10 | 40 %                 |
| proto/algebra.proto               | Rel                          |                          |                  8 |      21 | 38 %                 |
| proto/algebra.proto               | Expression                   |                          |                  4 |      12 | 33 %                 |
| proto/algebra.proto               | Expression                   | ReferenceSegment         |                  3 |       9 | 33 %                 |
| proto/algebra.proto               | FunctionArgument             |                          |                  1 |       3 | 33 %                 |
| proto/algebra.proto               | ComparisonJoinKey            |                          |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | ComparisonJoinKey            | ComparisonType           |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | ConsistentPartitionWindowRel |                          |                  0 |       6 | 0 %                  |
| proto/algebra.proto               | ConsistentPartitionWindowRel | WindowRelFunction        |                  0 |       9 | 0 %                  |
| proto/algebra.proto               | DdlRel                       |                          |                  0 |       8 | 0 %                  |
| proto/algebra.proto               | ExchangeRel                  |                          |                  0 |      10 | 0 %                  |
| proto/algebra.proto               | ExchangeRel                  | ExchangeTarget           |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | ExchangeRel                  | MultiBucketExpression    |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | ExchangeRel                  | RoundRobin               |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | ExchangeRel                  | ScatterFields            |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | ExchangeRel                  | SingleBucketExpression   |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | ExpandRel                    |                          |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | ExpandRel                    | ExpandField              |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | ExpandRel                    | SwitchingField           |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | EmbeddedFunction         |                  0 |       8 | 0 %                  |
| proto/algebra.proto               | Expression                   | Enum                     |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | IfThen                   |                  0 |       4 | 0 %                  |
| proto/algebra.proto               | Expression                   | MaskExpression           |                  0 |      20 | 0 %                  |
| proto/algebra.proto               | Expression                   | MultiOrList              |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | Expression                   | Nested                   |                  0 |      10 | 0 %                  |
| proto/algebra.proto               | Expression                   | SingularOrList           |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | Subquery                 |                  0 |      13 | 0 %                  |
| proto/algebra.proto               | Expression                   | SwitchExpression         |                  0 |       5 | 0 %                  |
| proto/algebra.proto               | Expression                   | WindowFunction           |                  0 |      18 | 0 %                  |
| proto/algebra.proto               | ExtensionLeafRel             |                          |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | ExtensionMultiRel            |                          |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | ExtensionObject              |                          |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | ExtensionSingleRel           |                          |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | FunctionOption               |                          |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | HashJoinRel                  |                          |                  0 |       9 | 0 %                  |
| proto/algebra.proto               | MergeJoinRel                 |                          |                  0 |       9 | 0 %                  |
| proto/algebra.proto               | NamedObjectWrite             |                          |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | NestedLoopJoinRel            |                          |                  0 |       6 | 0 %                  |
| proto/algebra.proto               | ReadRel                      | LocalFiles               |                  0 |      14 | 0 %                  |
| proto/algebra.proto               | ReadRel                      | VirtualTable             |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | ReferenceRel                 |                          |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | RelCommon                    | Hint                     |                  0 |       7 | 0 %                  |
| proto/algebra.proto               | SortField                    |                          |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | SortRel                      |                          |                  0 |       4 | 0 %                  |
| proto/algebra.proto               | WriteRel                     |                          |                  0 |       7 | 0 %                  |
| proto/capabilities.proto          | Capabilities                 |                          |                  0 |       3 | 0 %                  |
| proto/capabilities.proto          | Capabilities                 | SimpleExtension          |                  0 |       4 | 0 %                  |
| proto/extended_expression.proto   | ExpressionReference          |                          |                  0 |       3 | 0 %                  |
| proto/extended_expression.proto   | ExtendedExpression           |                          |                  0 |       7 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Aggregate                |                  0 |      12 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Argument                 |                  0 |       9 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Description              |                  0 |       2 | 0 %                  |
| proto/function.proto              | FunctionSignature            | FinalArgVariadic         |                  0 |       3 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Implementation           |                  0 |       2 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Scalar                   |                  0 |       9 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Window                   |                  0 |      13 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            |                          |                  0 |      26 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | IntegerOption            |                  0 |       2 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | IntegerParameter         |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | NullableInteger          |                  0 |       1 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedDecimal     |                  0 |       4 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedFixedBinary |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedFixedChar   |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedList        |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedMap         |                  0 |       4 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedNamedStruct |                  0 |       2 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedStruct      |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedUserDefined |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedVarChar     |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | TypeParameter            |                  0 |       2 | 0 %                  |
| proto/plan.proto                  | PlanVersion                  |                          |                  0 |       1 | 0 %                  |
| proto/type.proto                  | Type                         | FixedBinary              |                  0 |       3 | 0 %                  |
| proto/type.proto                  | Type                         | List                     |                  0 |       3 | 0 %                  |
| proto/type.proto                  | Type                         | Map                      |                  0 |       4 | 0 %                  |
| proto/type.proto                  | Type                         | Parameter                |                  0 |       6 | 0 %                  |
| proto/type.proto                  | Type                         | UserDefined              |                  0 |       4 | 0 %                  |
| proto/type.proto                  | Type                         | VarChar                  |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         |                          |                  0 |      32 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | BinaryOp                 |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionDecimal        |                  0 |       4 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionFixedBinary    |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionFixedChar      |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionList           |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionMap            |                  0 |       4 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionNamedStruct    |                  0 |       2 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionStruct         |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionUserDefined    |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionVarChar        |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | IfElse                   |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ReturnProgram            |                  0 |       4 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | UnaryOp                  |                  0 |       2 | 0 %                  |

| file                              | msg_lvl0                     | msg_lvl1                 | msg_lvl2             |   covered (number) |   total | covered (fraction)   |
|-----------------------------------|------------------------------|--------------------------|----------------------|--------------------|---------|----------------------|
| proto/algebra.proto               | AggregateRel                 | Grouping                 |                      |                  1 |       1 | 100 %                |
| proto/algebra.proto               | Expression                   | Cast                     |                      |                  3 |       3 | 100 %                |
| proto/algebra.proto               | Expression                   | Literal                  | Decimal              |                  3 |       3 | 100 %                |
| proto/algebra.proto               | Expression                   | Literal                  | IntervalYearToMonth  |                  2 |       2 | 100 %                |
| proto/algebra.proto               | Expression                   | ReferenceSegment         | StructField          |                  2 |       2 | 100 %                |
| proto/algebra.proto               | ProjectRel                   |                          |                      |                  4 |       4 | 100 %                |
| proto/algebra.proto               | ReadRel                      | ExtensionTable           |                      |                  1 |       1 | 100 %                |
| proto/algebra.proto               | RelCommon                    | Emit                     |                      |                  1 |       1 | 100 %                |
| proto/algebra.proto               | RelRoot                      |                          |                      |                  2 |       2 | 100 %                |
| proto/extensions/extensions.proto | AdvancedExtension            |                          |                      |                  2 |       2 | 100 %                |
| proto/extensions/extensions.proto | SimpleExtensionDeclaration   |                          |                      |                  3 |       3 | 100 %                |
| proto/extensions/extensions.proto | SimpleExtensionDeclaration   | ExtensionFunction        |                      |                  3 |       3 | 100 %                |
| proto/extensions/extensions.proto | SimpleExtensionDeclaration   | ExtensionType            |                      |                  3 |       3 | 100 %                |
| proto/extensions/extensions.proto | SimpleExtensionDeclaration   | ExtensionTypeVariation   |                      |                  3 |       3 | 100 %                |
| proto/extensions/extensions.proto | SimpleExtensionURI           |                          |                      |                  2 |       2 | 100 %                |
| proto/plan.proto                  | Plan                         |                          |                      |                  6 |       6 | 100 %                |
| proto/plan.proto                  | PlanRel                      |                          |                      |                  2 |       2 | 100 %                |
| proto/plan.proto                  | Version                      |                          |                      |                  5 |       5 | 100 %                |
| proto/type.proto                  | NamedStruct                  |                          |                      |                  2 |       2 | 100 %                |
| proto/algebra.proto               | AggregateRel                 |                          |                      |                  4 |       5 | 80 %                 |
| proto/algebra.proto               | FetchRel                     |                          |                      |                  4 |       5 | 80 %                 |
| proto/type.proto                  | Type                         |                          |                      |                 19 |      25 | 76 %                 |
| proto/algebra.proto               | CrossRel                     |                          |                      |                  3 |       4 | 75 %                 |
| proto/algebra.proto               | FilterRel                    |                          |                      |                  3 |       4 | 75 %                 |
| proto/algebra.proto               | SetRel                       |                          |                      |                  3 |       4 | 75 %                 |
| proto/type.proto                  | Type                         | Decimal                  |                      |                  3 |       4 | 75 %                 |
| proto/algebra.proto               | Expression                   | Literal                  | IntervalDayToSecond  |                  2 |       3 | 66 %                 |
| proto/type.proto                  | Type                         | FixedChar                |                      |                  2 |       3 | 66 %                 |
| proto/type.proto                  | Type                         | Struct                   |                      |                  2 |       3 | 66 %                 |
| proto/algebra.proto               | Expression                   | Literal                  |                      |                 18 |      29 | 62 %                 |
| proto/algebra.proto               | Expression                   | FieldReference           |                      |                  3 |       5 | 60 %                 |
| proto/algebra.proto               | Expression                   | ScalarFunction           |                      |                  3 |       5 | 60 %                 |
| proto/algebra.proto               | JoinRel                      |                          |                      |                  4 |       7 | 57 %                 |
| proto/algebra.proto               | AggregateFunction            |                          |                      |                  4 |       8 | 50 %                 |
| proto/algebra.proto               | AggregateRel                 | Measure                  |                      |                  1 |       2 | 50 %                 |
| proto/algebra.proto               | ReadRel                      | NamedTable               |                      |                  1 |       2 | 50 %                 |
| proto/algebra.proto               | RelCommon                    |                          |                      |                  2 |       4 | 50 %                 |
| proto/type.proto                  | Type                         | Binary                   |                      |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | Boolean                  |                      |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | Date                     |                      |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | FP32                     |                      |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | FP64                     |                      |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | I16                      |                      |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | I32                      |                      |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | I64                      |                      |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | I8                       |                      |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | IntervalDay              |                      |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | IntervalYear             |                      |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | String                   |                      |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | Time                     |                      |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | Timestamp                |                      |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | TimestampTZ              |                      |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | UUID                     |                      |                  1 |       2 | 50 %                 |
| proto/algebra.proto               | ReadRel                      |                          |                      |                  4 |      10 | 40 %                 |
| proto/algebra.proto               | Rel                          |                          |                      |                  8 |      21 | 38 %                 |
| proto/algebra.proto               | Expression                   |                          |                      |                  4 |      12 | 33 %                 |
| proto/algebra.proto               | Expression                   | ReferenceSegment         |                      |                  1 |       3 | 33 %                 |
| proto/algebra.proto               | FunctionArgument             |                          |                      |                  1 |       3 | 33 %                 |
| proto/algebra.proto               | ComparisonJoinKey            |                          |                      |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | ComparisonJoinKey            | ComparisonType           |                      |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | ConsistentPartitionWindowRel |                          |                      |                  0 |       6 | 0 %                  |
| proto/algebra.proto               | ConsistentPartitionWindowRel | WindowRelFunction        |                      |                  0 |       9 | 0 %                  |
| proto/algebra.proto               | DdlRel                       |                          |                      |                  0 |       8 | 0 %                  |
| proto/algebra.proto               | ExchangeRel                  |                          |                      |                  0 |      10 | 0 %                  |
| proto/algebra.proto               | ExchangeRel                  | ExchangeTarget           |                      |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | ExchangeRel                  | MultiBucketExpression    |                      |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | ExchangeRel                  | RoundRobin               |                      |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | ExchangeRel                  | ScatterFields            |                      |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | ExchangeRel                  | SingleBucketExpression   |                      |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | ExpandRel                    |                          |                      |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | ExpandRel                    | ExpandField              |                      |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | ExpandRel                    | SwitchingField           |                      |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | EmbeddedFunction         |                      |                  0 |       4 | 0 %                  |
| proto/algebra.proto               | Expression                   | EmbeddedFunction         | PythonPickleFunction |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | EmbeddedFunction         | WebAssemblyFunction  |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | Enum                     |                      |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | FieldReference           | OuterReference       |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | IfThen                   |                      |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | IfThen                   | IfClause             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | Literal                  | List                 |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | Literal                  | Map                  |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | Expression                   | Literal                  | Struct               |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | Literal                  | UserDefined          |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | Expression                   | Literal                  | VarChar              |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | MaskExpression           |                      |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | MaskExpression           | ListSelect           |                  0 |       7 | 0 %                  |
| proto/algebra.proto               | Expression                   | MaskExpression           | MapSelect            |                  0 |       5 | 0 %                  |
| proto/algebra.proto               | Expression                   | MaskExpression           | Select               |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | Expression                   | MaskExpression           | StructItem           |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | MaskExpression           | StructSelect         |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | MultiOrList              |                      |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | MultiOrList              | Record               |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | Nested                   |                      |                  0 |       5 | 0 %                  |
| proto/algebra.proto               | Expression                   | Nested                   | List                 |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | Nested                   | Map                  |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | Expression                   | Nested                   | Struct               |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | ReferenceSegment         | ListElement          |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | ReferenceSegment         | MapKey               |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | SingularOrList           |                      |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | Subquery                 |                      |                  0 |       4 | 0 %                  |
| proto/algebra.proto               | Expression                   | Subquery                 | InPredicate          |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | Subquery                 | Scalar               |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | Subquery                 | SetComparison        |                  0 |       4 | 0 %                  |
| proto/algebra.proto               | Expression                   | Subquery                 | SetPredicate         |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | SwitchExpression         |                      |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | Expression                   | SwitchExpression         | IfValue              |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | WindowFunction           |                      |                  0 |      12 | 0 %                  |
| proto/algebra.proto               | Expression                   | WindowFunction           | Bound                |                  0 |       6 | 0 %                  |
| proto/algebra.proto               | ExtensionLeafRel             |                          |                      |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | ExtensionMultiRel            |                          |                      |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | ExtensionObject              |                          |                      |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | ExtensionSingleRel           |                          |                      |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | FunctionOption               |                          |                      |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | HashJoinRel                  |                          |                      |                  0 |       9 | 0 %                  |
| proto/algebra.proto               | MergeJoinRel                 |                          |                      |                  0 |       9 | 0 %                  |
| proto/algebra.proto               | NamedObjectWrite             |                          |                      |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | NestedLoopJoinRel            |                          |                      |                  0 |       6 | 0 %                  |
| proto/algebra.proto               | ReadRel                      | LocalFiles               |                      |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | ReadRel                      | LocalFiles               | FileOrFiles          |                  0 |      12 | 0 %                  |
| proto/algebra.proto               | ReadRel                      | VirtualTable             |                      |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | ReferenceRel                 |                          |                      |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | RelCommon                    | Hint                     |                      |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | RelCommon                    | Hint                     | RuntimeConstraint    |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | RelCommon                    | Hint                     | Stats                |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | SortField                    |                          |                      |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | SortRel                      |                          |                      |                  0 |       4 | 0 %                  |
| proto/algebra.proto               | WriteRel                     |                          |                      |                  0 |       7 | 0 %                  |
| proto/capabilities.proto          | Capabilities                 |                          |                      |                  0 |       3 | 0 %                  |
| proto/capabilities.proto          | Capabilities                 | SimpleExtension          |                      |                  0 |       4 | 0 %                  |
| proto/extended_expression.proto   | ExpressionReference          |                          |                      |                  0 |       3 | 0 %                  |
| proto/extended_expression.proto   | ExtendedExpression           |                          |                      |                  0 |       7 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Aggregate                |                      |                  0 |      12 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Argument                 |                      |                  0 |       4 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Argument                 | EnumArgument         |                  0 |       2 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Argument                 | TypeArgument         |                  0 |       1 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Argument                 | ValueArgument        |                  0 |       2 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Description              |                      |                  0 |       2 | 0 %                  |
| proto/function.proto              | FunctionSignature            | FinalArgVariadic         |                      |                  0 |       3 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Implementation           |                      |                  0 |       2 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Scalar                   |                      |                  0 |       9 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Window                   |                      |                  0 |      13 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            |                          |                      |                  0 |      26 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | IntegerOption            |                      |                  0 |       2 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | IntegerParameter         |                      |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | NullableInteger          |                      |                  0 |       1 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedDecimal     |                      |                  0 |       4 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedFixedBinary |                      |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedFixedChar   |                      |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedList        |                      |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedMap         |                      |                  0 |       4 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedNamedStruct |                      |                  0 |       2 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedStruct      |                      |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedUserDefined |                      |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedVarChar     |                      |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | TypeParameter            |                      |                  0 |       2 | 0 %                  |
| proto/plan.proto                  | PlanVersion                  |                          |                      |                  0 |       1 | 0 %                  |
| proto/type.proto                  | Type                         | FixedBinary              |                      |                  0 |       3 | 0 %                  |
| proto/type.proto                  | Type                         | List                     |                      |                  0 |       3 | 0 %                  |
| proto/type.proto                  | Type                         | Map                      |                      |                  0 |       4 | 0 %                  |
| proto/type.proto                  | Type                         | Parameter                |                      |                  0 |       6 | 0 %                  |
| proto/type.proto                  | Type                         | UserDefined              |                      |                  0 |       4 | 0 %                  |
| proto/type.proto                  | Type                         | VarChar                  |                      |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         |                          |                      |                  0 |      32 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | BinaryOp                 |                      |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionDecimal        |                      |                  0 |       4 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionFixedBinary    |                      |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionFixedChar      |                      |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionList           |                      |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionMap            |                      |                  0 |       4 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionNamedStruct    |                      |                  0 |       2 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionStruct         |                      |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionUserDefined    |                      |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionVarChar        |                      |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | IfElse                   |                      |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ReturnProgram            |                      |                  0 |       2 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ReturnProgram            | Assignment           |                  0 |       2 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | UnaryOp                  |                      |                  0 |       2 | 0 %                  |

| file                              | msg_lvl0                     | msg_lvl1                 | msg_lvl2             | msg_lvl3         |   covered (number) |   total | covered (fraction)   |
|-----------------------------------|------------------------------|--------------------------|----------------------|------------------|--------------------|---------|----------------------|
| proto/algebra.proto               | AggregateRel                 | Grouping                 |                      |                  |                  1 |       1 | 100 %                |
| proto/algebra.proto               | Expression                   | Cast                     |                      |                  |                  3 |       3 | 100 %                |
| proto/algebra.proto               | Expression                   | Literal                  | Decimal              |                  |                  3 |       3 | 100 %                |
| proto/algebra.proto               | Expression                   | Literal                  | IntervalYearToMonth  |                  |                  2 |       2 | 100 %                |
| proto/algebra.proto               | Expression                   | ReferenceSegment         | StructField          |                  |                  2 |       2 | 100 %                |
| proto/algebra.proto               | ProjectRel                   |                          |                      |                  |                  4 |       4 | 100 %                |
| proto/algebra.proto               | ReadRel                      | ExtensionTable           |                      |                  |                  1 |       1 | 100 %                |
| proto/algebra.proto               | RelCommon                    | Emit                     |                      |                  |                  1 |       1 | 100 %                |
| proto/algebra.proto               | RelRoot                      |                          |                      |                  |                  2 |       2 | 100 %                |
| proto/extensions/extensions.proto | AdvancedExtension            |                          |                      |                  |                  2 |       2 | 100 %                |
| proto/extensions/extensions.proto | SimpleExtensionDeclaration   |                          |                      |                  |                  3 |       3 | 100 %                |
| proto/extensions/extensions.proto | SimpleExtensionDeclaration   | ExtensionFunction        |                      |                  |                  3 |       3 | 100 %                |
| proto/extensions/extensions.proto | SimpleExtensionDeclaration   | ExtensionType            |                      |                  |                  3 |       3 | 100 %                |
| proto/extensions/extensions.proto | SimpleExtensionDeclaration   | ExtensionTypeVariation   |                      |                  |                  3 |       3 | 100 %                |
| proto/extensions/extensions.proto | SimpleExtensionURI           |                          |                      |                  |                  2 |       2 | 100 %                |
| proto/plan.proto                  | Plan                         |                          |                      |                  |                  6 |       6 | 100 %                |
| proto/plan.proto                  | PlanRel                      |                          |                      |                  |                  2 |       2 | 100 %                |
| proto/plan.proto                  | Version                      |                          |                      |                  |                  5 |       5 | 100 %                |
| proto/type.proto                  | NamedStruct                  |                          |                      |                  |                  2 |       2 | 100 %                |
| proto/algebra.proto               | AggregateRel                 |                          |                      |                  |                  4 |       5 | 80 %                 |
| proto/algebra.proto               | FetchRel                     |                          |                      |                  |                  4 |       5 | 80 %                 |
| proto/type.proto                  | Type                         |                          |                      |                  |                 19 |      25 | 76 %                 |
| proto/algebra.proto               | CrossRel                     |                          |                      |                  |                  3 |       4 | 75 %                 |
| proto/algebra.proto               | FilterRel                    |                          |                      |                  |                  3 |       4 | 75 %                 |
| proto/algebra.proto               | SetRel                       |                          |                      |                  |                  3 |       4 | 75 %                 |
| proto/type.proto                  | Type                         | Decimal                  |                      |                  |                  3 |       4 | 75 %                 |
| proto/algebra.proto               | Expression                   | Literal                  | IntervalDayToSecond  |                  |                  2 |       3 | 66 %                 |
| proto/type.proto                  | Type                         | FixedChar                |                      |                  |                  2 |       3 | 66 %                 |
| proto/type.proto                  | Type                         | Struct                   |                      |                  |                  2 |       3 | 66 %                 |
| proto/algebra.proto               | Expression                   | Literal                  |                      |                  |                 18 |      29 | 62 %                 |
| proto/algebra.proto               | Expression                   | FieldReference           |                      |                  |                  3 |       5 | 60 %                 |
| proto/algebra.proto               | Expression                   | ScalarFunction           |                      |                  |                  3 |       5 | 60 %                 |
| proto/algebra.proto               | JoinRel                      |                          |                      |                  |                  4 |       7 | 57 %                 |
| proto/algebra.proto               | AggregateFunction            |                          |                      |                  |                  4 |       8 | 50 %                 |
| proto/algebra.proto               | AggregateRel                 | Measure                  |                      |                  |                  1 |       2 | 50 %                 |
| proto/algebra.proto               | ReadRel                      | NamedTable               |                      |                  |                  1 |       2 | 50 %                 |
| proto/algebra.proto               | RelCommon                    |                          |                      |                  |                  2 |       4 | 50 %                 |
| proto/type.proto                  | Type                         | Binary                   |                      |                  |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | Boolean                  |                      |                  |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | Date                     |                      |                  |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | FP32                     |                      |                  |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | FP64                     |                      |                  |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | I16                      |                      |                  |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | I32                      |                      |                  |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | I64                      |                      |                  |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | I8                       |                      |                  |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | IntervalDay              |                      |                  |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | IntervalYear             |                      |                  |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | String                   |                      |                  |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | Time                     |                      |                  |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | Timestamp                |                      |                  |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | TimestampTZ              |                      |                  |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | UUID                     |                      |                  |                  1 |       2 | 50 %                 |
| proto/algebra.proto               | ReadRel                      |                          |                      |                  |                  4 |      10 | 40 %                 |
| proto/algebra.proto               | Rel                          |                          |                      |                  |                  8 |      21 | 38 %                 |
| proto/algebra.proto               | Expression                   |                          |                      |                  |                  4 |      12 | 33 %                 |
| proto/algebra.proto               | Expression                   | ReferenceSegment         |                      |                  |                  1 |       3 | 33 %                 |
| proto/algebra.proto               | FunctionArgument             |                          |                      |                  |                  1 |       3 | 33 %                 |
| proto/algebra.proto               | ComparisonJoinKey            |                          |                      |                  |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | ComparisonJoinKey            | ComparisonType           |                      |                  |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | ConsistentPartitionWindowRel |                          |                      |                  |                  0 |       6 | 0 %                  |
| proto/algebra.proto               | ConsistentPartitionWindowRel | WindowRelFunction        |                      |                  |                  0 |       9 | 0 %                  |
| proto/algebra.proto               | DdlRel                       |                          |                      |                  |                  0 |       8 | 0 %                  |
| proto/algebra.proto               | ExchangeRel                  |                          |                      |                  |                  0 |      10 | 0 %                  |
| proto/algebra.proto               | ExchangeRel                  | ExchangeTarget           |                      |                  |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | ExchangeRel                  | MultiBucketExpression    |                      |                  |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | ExchangeRel                  | RoundRobin               |                      |                  |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | ExchangeRel                  | ScatterFields            |                      |                  |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | ExchangeRel                  | SingleBucketExpression   |                      |                  |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | ExpandRel                    |                          |                      |                  |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | ExpandRel                    | ExpandField              |                      |                  |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | ExpandRel                    | SwitchingField           |                      |                  |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | EmbeddedFunction         |                      |                  |                  0 |       4 | 0 %                  |
| proto/algebra.proto               | Expression                   | EmbeddedFunction         | PythonPickleFunction |                  |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | EmbeddedFunction         | WebAssemblyFunction  |                  |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | Enum                     |                      |                  |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | FieldReference           | OuterReference       |                  |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | IfThen                   |                      |                  |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | IfThen                   | IfClause             |                  |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | Literal                  | List                 |                  |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | Literal                  | Map                  |                  |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | Literal                  | Map                  | KeyValue         |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | Literal                  | Struct               |                  |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | Literal                  | UserDefined          |                  |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | Expression                   | Literal                  | VarChar              |                  |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | MaskExpression           |                      |                  |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | MaskExpression           | ListSelect           |                  |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | MaskExpression           | ListSelect           | ListSelectItem   |                  0 |       5 | 0 %                  |
| proto/algebra.proto               | Expression                   | MaskExpression           | MapSelect            |                  |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | Expression                   | MaskExpression           | MapSelect            | MapKey           |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | MaskExpression           | MapSelect            | MapKeyExpression |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | MaskExpression           | Select               |                  |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | Expression                   | MaskExpression           | StructItem           |                  |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | MaskExpression           | StructSelect         |                  |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | MultiOrList              |                      |                  |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | MultiOrList              | Record               |                  |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | Nested                   |                      |                  |                  0 |       5 | 0 %                  |
| proto/algebra.proto               | Expression                   | Nested                   | List                 |                  |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | Nested                   | Map                  |                  |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | Nested                   | Map                  | KeyValue         |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | Nested                   | Struct               |                  |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | ReferenceSegment         | ListElement          |                  |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | ReferenceSegment         | MapKey               |                  |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | SingularOrList           |                      |                  |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | Subquery                 |                      |                  |                  0 |       4 | 0 %                  |
| proto/algebra.proto               | Expression                   | Subquery                 | InPredicate          |                  |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | Subquery                 | Scalar               |                  |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | Subquery                 | SetComparison        |                  |                  0 |       4 | 0 %                  |
| proto/algebra.proto               | Expression                   | Subquery                 | SetPredicate         |                  |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | SwitchExpression         |                      |                  |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | Expression                   | SwitchExpression         | IfValue              |                  |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | WindowFunction           |                      |                  |                  0 |      12 | 0 %                  |
| proto/algebra.proto               | Expression                   | WindowFunction           | Bound                |                  |                  0 |       4 | 0 %                  |
| proto/algebra.proto               | Expression                   | WindowFunction           | Bound                | Following        |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | WindowFunction           | Bound                | Preceding        |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | ExtensionLeafRel             |                          |                      |                  |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | ExtensionMultiRel            |                          |                      |                  |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | ExtensionObject              |                          |                      |                  |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | ExtensionSingleRel           |                          |                      |                  |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | FunctionOption               |                          |                      |                  |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | HashJoinRel                  |                          |                      |                  |                  0 |       9 | 0 %                  |
| proto/algebra.proto               | MergeJoinRel                 |                          |                      |                  |                  0 |       9 | 0 %                  |
| proto/algebra.proto               | NamedObjectWrite             |                          |                      |                  |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | NestedLoopJoinRel            |                          |                      |                  |                  0 |       6 | 0 %                  |
| proto/algebra.proto               | ReadRel                      | LocalFiles               |                      |                  |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | ReadRel                      | LocalFiles               | FileOrFiles          |                  |                  0 |      12 | 0 %                  |
| proto/algebra.proto               | ReadRel                      | VirtualTable             |                      |                  |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | ReferenceRel                 |                          |                      |                  |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | RelCommon                    | Hint                     |                      |                  |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | RelCommon                    | Hint                     | RuntimeConstraint    |                  |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | RelCommon                    | Hint                     | Stats                |                  |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | SortField                    |                          |                      |                  |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | SortRel                      |                          |                      |                  |                  0 |       4 | 0 %                  |
| proto/algebra.proto               | WriteRel                     |                          |                      |                  |                  0 |       7 | 0 %                  |
| proto/capabilities.proto          | Capabilities                 |                          |                      |                  |                  0 |       3 | 0 %                  |
| proto/capabilities.proto          | Capabilities                 | SimpleExtension          |                      |                  |                  0 |       4 | 0 %                  |
| proto/extended_expression.proto   | ExpressionReference          |                          |                      |                  |                  0 |       3 | 0 %                  |
| proto/extended_expression.proto   | ExtendedExpression           |                          |                      |                  |                  0 |       7 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Aggregate                |                      |                  |                  0 |      12 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Argument                 |                      |                  |                  0 |       4 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Argument                 | EnumArgument         |                  |                  0 |       2 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Argument                 | TypeArgument         |                  |                  0 |       1 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Argument                 | ValueArgument        |                  |                  0 |       2 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Description              |                      |                  |                  0 |       2 | 0 %                  |
| proto/function.proto              | FunctionSignature            | FinalArgVariadic         |                      |                  |                  0 |       3 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Implementation           |                      |                  |                  0 |       2 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Scalar                   |                      |                  |                  0 |       9 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Window                   |                      |                  |                  0 |      13 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            |                          |                      |                  |                  0 |      26 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | IntegerOption            |                      |                  |                  0 |       2 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | IntegerParameter         |                      |                  |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | NullableInteger          |                      |                  |                  0 |       1 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedDecimal     |                      |                  |                  0 |       4 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedFixedBinary |                      |                  |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedFixedChar   |                      |                  |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedList        |                      |                  |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedMap         |                      |                  |                  0 |       4 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedNamedStruct |                      |                  |                  0 |       2 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedStruct      |                      |                  |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedUserDefined |                      |                  |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedVarChar     |                      |                  |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | TypeParameter            |                      |                  |                  0 |       2 | 0 %                  |
| proto/plan.proto                  | PlanVersion                  |                          |                      |                  |                  0 |       1 | 0 %                  |
| proto/type.proto                  | Type                         | FixedBinary              |                      |                  |                  0 |       3 | 0 %                  |
| proto/type.proto                  | Type                         | List                     |                      |                  |                  0 |       3 | 0 %                  |
| proto/type.proto                  | Type                         | Map                      |                      |                  |                  0 |       4 | 0 %                  |
| proto/type.proto                  | Type                         | Parameter                |                      |                  |                  0 |       6 | 0 %                  |
| proto/type.proto                  | Type                         | UserDefined              |                      |                  |                  0 |       4 | 0 %                  |
| proto/type.proto                  | Type                         | VarChar                  |                      |                  |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         |                          |                      |                  |                  0 |      32 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | BinaryOp                 |                      |                  |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionDecimal        |                      |                  |                  0 |       4 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionFixedBinary    |                      |                  |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionFixedChar      |                      |                  |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionList           |                      |                  |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionMap            |                      |                  |                  0 |       4 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionNamedStruct    |                      |                  |                  0 |       2 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionStruct         |                      |                  |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionUserDefined    |                      |                  |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionVarChar        |                      |                  |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | IfElse                   |                      |                  |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ReturnProgram            |                      |                  |                  0 |       2 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ReturnProgram            | Assignment           |                  |                  0 |       2 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | UnaryOp                  |                      |                  |                  0 |       2 | 0 %                  |

| file                              | msg_lvl0                     | msg_lvl1                 | msg_lvl2             | msg_lvl3         | msg_lvl4    |   covered (number) |   total | covered (fraction)   |
|-----------------------------------|------------------------------|--------------------------|----------------------|------------------|-------------|--------------------|---------|----------------------|
| proto/algebra.proto               | AggregateRel                 | Grouping                 |                      |                  |             |                  1 |       1 | 100 %                |
| proto/algebra.proto               | Expression                   | Cast                     |                      |                  |             |                  3 |       3 | 100 %                |
| proto/algebra.proto               | Expression                   | Literal                  | Decimal              |                  |             |                  3 |       3 | 100 %                |
| proto/algebra.proto               | Expression                   | Literal                  | IntervalYearToMonth  |                  |             |                  2 |       2 | 100 %                |
| proto/algebra.proto               | Expression                   | ReferenceSegment         | StructField          |                  |             |                  2 |       2 | 100 %                |
| proto/algebra.proto               | ProjectRel                   |                          |                      |                  |             |                  4 |       4 | 100 %                |
| proto/algebra.proto               | ReadRel                      | ExtensionTable           |                      |                  |             |                  1 |       1 | 100 %                |
| proto/algebra.proto               | RelCommon                    | Emit                     |                      |                  |             |                  1 |       1 | 100 %                |
| proto/algebra.proto               | RelRoot                      |                          |                      |                  |             |                  2 |       2 | 100 %                |
| proto/extensions/extensions.proto | AdvancedExtension            |                          |                      |                  |             |                  2 |       2 | 100 %                |
| proto/extensions/extensions.proto | SimpleExtensionDeclaration   |                          |                      |                  |             |                  3 |       3 | 100 %                |
| proto/extensions/extensions.proto | SimpleExtensionDeclaration   | ExtensionFunction        |                      |                  |             |                  3 |       3 | 100 %                |
| proto/extensions/extensions.proto | SimpleExtensionDeclaration   | ExtensionType            |                      |                  |             |                  3 |       3 | 100 %                |
| proto/extensions/extensions.proto | SimpleExtensionDeclaration   | ExtensionTypeVariation   |                      |                  |             |                  3 |       3 | 100 %                |
| proto/extensions/extensions.proto | SimpleExtensionURI           |                          |                      |                  |             |                  2 |       2 | 100 %                |
| proto/plan.proto                  | Plan                         |                          |                      |                  |             |                  6 |       6 | 100 %                |
| proto/plan.proto                  | PlanRel                      |                          |                      |                  |             |                  2 |       2 | 100 %                |
| proto/plan.proto                  | Version                      |                          |                      |                  |             |                  5 |       5 | 100 %                |
| proto/type.proto                  | NamedStruct                  |                          |                      |                  |             |                  2 |       2 | 100 %                |
| proto/algebra.proto               | AggregateRel                 |                          |                      |                  |             |                  4 |       5 | 80 %                 |
| proto/algebra.proto               | FetchRel                     |                          |                      |                  |             |                  4 |       5 | 80 %                 |
| proto/type.proto                  | Type                         |                          |                      |                  |             |                 19 |      25 | 76 %                 |
| proto/algebra.proto               | CrossRel                     |                          |                      |                  |             |                  3 |       4 | 75 %                 |
| proto/algebra.proto               | FilterRel                    |                          |                      |                  |             |                  3 |       4 | 75 %                 |
| proto/algebra.proto               | SetRel                       |                          |                      |                  |             |                  3 |       4 | 75 %                 |
| proto/type.proto                  | Type                         | Decimal                  |                      |                  |             |                  3 |       4 | 75 %                 |
| proto/algebra.proto               | Expression                   | Literal                  | IntervalDayToSecond  |                  |             |                  2 |       3 | 66 %                 |
| proto/type.proto                  | Type                         | FixedChar                |                      |                  |             |                  2 |       3 | 66 %                 |
| proto/type.proto                  | Type                         | Struct                   |                      |                  |             |                  2 |       3 | 66 %                 |
| proto/algebra.proto               | Expression                   | Literal                  |                      |                  |             |                 18 |      29 | 62 %                 |
| proto/algebra.proto               | Expression                   | FieldReference           |                      |                  |             |                  3 |       5 | 60 %                 |
| proto/algebra.proto               | Expression                   | ScalarFunction           |                      |                  |             |                  3 |       5 | 60 %                 |
| proto/algebra.proto               | JoinRel                      |                          |                      |                  |             |                  4 |       7 | 57 %                 |
| proto/algebra.proto               | AggregateFunction            |                          |                      |                  |             |                  4 |       8 | 50 %                 |
| proto/algebra.proto               | AggregateRel                 | Measure                  |                      |                  |             |                  1 |       2 | 50 %                 |
| proto/algebra.proto               | ReadRel                      | NamedTable               |                      |                  |             |                  1 |       2 | 50 %                 |
| proto/algebra.proto               | RelCommon                    |                          |                      |                  |             |                  2 |       4 | 50 %                 |
| proto/type.proto                  | Type                         | Binary                   |                      |                  |             |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | Boolean                  |                      |                  |             |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | Date                     |                      |                  |             |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | FP32                     |                      |                  |             |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | FP64                     |                      |                  |             |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | I16                      |                      |                  |             |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | I32                      |                      |                  |             |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | I64                      |                      |                  |             |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | I8                       |                      |                  |             |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | IntervalDay              |                      |                  |             |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | IntervalYear             |                      |                  |             |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | String                   |                      |                  |             |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | Time                     |                      |                  |             |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | Timestamp                |                      |                  |             |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | TimestampTZ              |                      |                  |             |                  1 |       2 | 50 %                 |
| proto/type.proto                  | Type                         | UUID                     |                      |                  |             |                  1 |       2 | 50 %                 |
| proto/algebra.proto               | ReadRel                      |                          |                      |                  |             |                  4 |      10 | 40 %                 |
| proto/algebra.proto               | Rel                          |                          |                      |                  |             |                  8 |      21 | 38 %                 |
| proto/algebra.proto               | Expression                   |                          |                      |                  |             |                  4 |      12 | 33 %                 |
| proto/algebra.proto               | Expression                   | ReferenceSegment         |                      |                  |             |                  1 |       3 | 33 %                 |
| proto/algebra.proto               | FunctionArgument             |                          |                      |                  |             |                  1 |       3 | 33 %                 |
| proto/algebra.proto               | ComparisonJoinKey            |                          |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | ComparisonJoinKey            | ComparisonType           |                      |                  |             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | ConsistentPartitionWindowRel |                          |                      |                  |             |                  0 |       6 | 0 %                  |
| proto/algebra.proto               | ConsistentPartitionWindowRel | WindowRelFunction        |                      |                  |             |                  0 |       9 | 0 %                  |
| proto/algebra.proto               | DdlRel                       |                          |                      |                  |             |                  0 |       8 | 0 %                  |
| proto/algebra.proto               | ExchangeRel                  |                          |                      |                  |             |                  0 |      10 | 0 %                  |
| proto/algebra.proto               | ExchangeRel                  | ExchangeTarget           |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | ExchangeRel                  | MultiBucketExpression    |                      |                  |             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | ExchangeRel                  | RoundRobin               |                      |                  |             |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | ExchangeRel                  | ScatterFields            |                      |                  |             |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | ExchangeRel                  | SingleBucketExpression   |                      |                  |             |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | ExpandRel                    |                          |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | ExpandRel                    | ExpandField              |                      |                  |             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | ExpandRel                    | SwitchingField           |                      |                  |             |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | EmbeddedFunction         |                      |                  |             |                  0 |       4 | 0 %                  |
| proto/algebra.proto               | Expression                   | EmbeddedFunction         | PythonPickleFunction |                  |             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | EmbeddedFunction         | WebAssemblyFunction  |                  |             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | Enum                     |                      |                  |             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | FieldReference           | OuterReference       |                  |             |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | IfThen                   |                      |                  |             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | IfThen                   | IfClause             |                  |             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | Literal                  | List                 |                  |             |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | Literal                  | Map                  |                  |             |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | Literal                  | Map                  | KeyValue         |             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | Literal                  | Struct               |                  |             |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | Literal                  | UserDefined          |                  |             |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | Expression                   | Literal                  | VarChar              |                  |             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | MaskExpression           |                      |                  |             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | MaskExpression           | ListSelect           |                  |             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | MaskExpression           | ListSelect           | ListSelectItem   |             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | MaskExpression           | ListSelect           | ListSelectItem   | ListElement |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | MaskExpression           | ListSelect           | ListSelectItem   | ListSlice   |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | MaskExpression           | MapSelect            |                  |             |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | Expression                   | MaskExpression           | MapSelect            | MapKey           |             |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | MaskExpression           | MapSelect            | MapKeyExpression |             |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | MaskExpression           | Select               |                  |             |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | Expression                   | MaskExpression           | StructItem           |                  |             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | MaskExpression           | StructSelect         |                  |             |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | MultiOrList              |                      |                  |             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | MultiOrList              | Record               |                  |             |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | Nested                   |                      |                  |             |                  0 |       5 | 0 %                  |
| proto/algebra.proto               | Expression                   | Nested                   | List                 |                  |             |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | Nested                   | Map                  |                  |             |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | Nested                   | Map                  | KeyValue         |             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | Nested                   | Struct               |                  |             |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | ReferenceSegment         | ListElement          |                  |             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | ReferenceSegment         | MapKey               |                  |             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | SingularOrList           |                      |                  |             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | Subquery                 |                      |                  |             |                  0 |       4 | 0 %                  |
| proto/algebra.proto               | Expression                   | Subquery                 | InPredicate          |                  |             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | Subquery                 | Scalar               |                  |             |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | Subquery                 | SetComparison        |                  |             |                  0 |       4 | 0 %                  |
| proto/algebra.proto               | Expression                   | Subquery                 | SetPredicate         |                  |             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | SwitchExpression         |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | Expression                   | SwitchExpression         | IfValue              |                  |             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | Expression                   | WindowFunction           |                      |                  |             |                  0 |      12 | 0 %                  |
| proto/algebra.proto               | Expression                   | WindowFunction           | Bound                |                  |             |                  0 |       4 | 0 %                  |
| proto/algebra.proto               | Expression                   | WindowFunction           | Bound                | Following        |             |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | Expression                   | WindowFunction           | Bound                | Preceding        |             |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | ExtensionLeafRel             |                          |                      |                  |             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | ExtensionMultiRel            |                          |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | ExtensionObject              |                          |                      |                  |             |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | ExtensionSingleRel           |                          |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | FunctionOption               |                          |                      |                  |             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | HashJoinRel                  |                          |                      |                  |             |                  0 |       9 | 0 %                  |
| proto/algebra.proto               | MergeJoinRel                 |                          |                      |                  |             |                  0 |       9 | 0 %                  |
| proto/algebra.proto               | NamedObjectWrite             |                          |                      |                  |             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | NestedLoopJoinRel            |                          |                      |                  |             |                  0 |       6 | 0 %                  |
| proto/algebra.proto               | ReadRel                      | LocalFiles               |                      |                  |             |                  0 |       2 | 0 %                  |
| proto/algebra.proto               | ReadRel                      | LocalFiles               | FileOrFiles          |                  |             |                  0 |      12 | 0 %                  |
| proto/algebra.proto               | ReadRel                      | VirtualTable             |                      |                  |             |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | ReferenceRel                 |                          |                      |                  |             |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | RelCommon                    | Hint                     |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | RelCommon                    | Hint                     | RuntimeConstraint    |                  |             |                  0 |       1 | 0 %                  |
| proto/algebra.proto               | RelCommon                    | Hint                     | Stats                |                  |             |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | SortField                    |                          |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/algebra.proto               | SortRel                      |                          |                      |                  |             |                  0 |       4 | 0 %                  |
| proto/algebra.proto               | WriteRel                     |                          |                      |                  |             |                  0 |       7 | 0 %                  |
| proto/capabilities.proto          | Capabilities                 |                          |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/capabilities.proto          | Capabilities                 | SimpleExtension          |                      |                  |             |                  0 |       4 | 0 %                  |
| proto/extended_expression.proto   | ExpressionReference          |                          |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/extended_expression.proto   | ExtendedExpression           |                          |                      |                  |             |                  0 |       7 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Aggregate                |                      |                  |             |                  0 |      12 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Argument                 |                      |                  |             |                  0 |       4 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Argument                 | EnumArgument         |                  |             |                  0 |       2 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Argument                 | TypeArgument         |                  |             |                  0 |       1 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Argument                 | ValueArgument        |                  |             |                  0 |       2 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Description              |                      |                  |             |                  0 |       2 | 0 %                  |
| proto/function.proto              | FunctionSignature            | FinalArgVariadic         |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Implementation           |                      |                  |             |                  0 |       2 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Scalar                   |                      |                  |             |                  0 |       9 | 0 %                  |
| proto/function.proto              | FunctionSignature            | Window                   |                      |                  |             |                  0 |      13 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            |                          |                      |                  |             |                  0 |      26 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | IntegerOption            |                      |                  |             |                  0 |       2 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | IntegerParameter         |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | NullableInteger          |                      |                  |             |                  0 |       1 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedDecimal     |                      |                  |             |                  0 |       4 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedFixedBinary |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedFixedChar   |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedList        |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedMap         |                      |                  |             |                  0 |       4 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedNamedStruct |                      |                  |             |                  0 |       2 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedStruct      |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedUserDefined |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | ParameterizedVarChar     |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/parameterized_types.proto   | ParameterizedType            | TypeParameter            |                      |                  |             |                  0 |       2 | 0 %                  |
| proto/plan.proto                  | PlanVersion                  |                          |                      |                  |             |                  0 |       1 | 0 %                  |
| proto/type.proto                  | Type                         | FixedBinary              |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/type.proto                  | Type                         | List                     |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/type.proto                  | Type                         | Map                      |                      |                  |             |                  0 |       4 | 0 %                  |
| proto/type.proto                  | Type                         | Parameter                |                      |                  |             |                  0 |       6 | 0 %                  |
| proto/type.proto                  | Type                         | UserDefined              |                      |                  |             |                  0 |       4 | 0 %                  |
| proto/type.proto                  | Type                         | VarChar                  |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         |                          |                      |                  |             |                  0 |      32 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | BinaryOp                 |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionDecimal        |                      |                  |             |                  0 |       4 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionFixedBinary    |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionFixedChar      |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionList           |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionMap            |                      |                  |             |                  0 |       4 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionNamedStruct    |                      |                  |             |                  0 |       2 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionStruct         |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionUserDefined    |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ExpressionVarChar        |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | IfElse                   |                      |                  |             |                  0 |       3 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ReturnProgram            |                      |                  |             |                  0 |       2 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | ReturnProgram            | Assignment           |                  |             |                  0 |       2 | 0 %                  |
| proto/type_expressions.proto      | DerivationExpression         | UnaryOp                  |                      |                  |             |                  0 |       2 | 0 %                  |

