// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use crate::PhysicalExpr;
use crate::expressions::Literal;
use arrow::array::{AsArray, BooleanArray};
use arrow::datatypes::{DataType, Schema};
use arrow::record_batch::RecordBatch;
use datafusion_common::{Result, ScalarValue, assert_or_internal_err};
use datafusion_expr::{ColumnarValue, Operator};
use datafusion_physical_expr_common::datum::apply_cmp;
use regex::{Regex, RegexBuilder};
use std::fmt;
use std::hash::Hash;
use std::sync::{Arc, OnceLock};

// Like expression
pub struct LikeExpr {
    negated: bool,
    case_insensitive: bool,
    expr: Arc<dyn PhysicalExpr>,
    pattern: Arc<dyn PhysicalExpr>,
    literal_regex: OnceLock<Option<Box<(DataType, Regex)>>>,
}

impl fmt::Debug for LikeExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LikeExpr")
            .field("negated", &self.negated)
            .field("case_insensitive", &self.case_insensitive)
            .field("expr", &self.expr)
            .field("pattern", &self.pattern)
            .finish()
    }
}

// Manually derive PartialEq and Hash to work around https://github.com/rust-lang/rust/issues/78808
impl PartialEq for LikeExpr {
    fn eq(&self, other: &Self) -> bool {
        self.negated == other.negated
            && self.case_insensitive == other.case_insensitive
            && self.expr.eq(&other.expr)
            && self.pattern.eq(&other.pattern)
    }
}

impl Eq for LikeExpr {}

impl Hash for LikeExpr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.negated.hash(state);
        self.case_insensitive.hash(state);
        self.expr.hash(state);
        self.pattern.hash(state);
    }
}

impl LikeExpr {
    pub fn new(
        negated: bool,
        case_insensitive: bool,
        expr: Arc<dyn PhysicalExpr>,
        pattern: Arc<dyn PhysicalExpr>,
    ) -> Self {
        Self {
            negated,
            case_insensitive,
            expr,
            pattern,
            literal_regex: OnceLock::new(),
        }
    }

    fn literal_regex(&self) -> Option<&(DataType, Regex)> {
        self.literal_regex
            .get_or_init(|| {
                let value = self.pattern.downcast_ref::<Literal>()?.value();
                let (data_type, pattern) = match value {
                    ScalarValue::Utf8(Some(pattern)) => (DataType::Utf8, pattern),
                    ScalarValue::LargeUtf8(Some(pattern)) => {
                        (DataType::LargeUtf8, pattern)
                    }
                    ScalarValue::Utf8View(Some(pattern)) => (DataType::Utf8View, pattern),
                    _ => return None,
                };
                Some(Box::new((data_type, compile_like_regex(pattern)?)))
            })
            .as_deref()
    }

    /// Is negated
    pub fn negated(&self) -> bool {
        self.negated
    }

    /// Is case insensitive
    pub fn case_insensitive(&self) -> bool {
        self.case_insensitive
    }

    /// Input expression
    pub fn expr(&self) -> &Arc<dyn PhysicalExpr> {
        &self.expr
    }

    /// Pattern expression
    pub fn pattern(&self) -> &Arc<dyn PhysicalExpr> {
        &self.pattern
    }

    /// Operator name
    fn op_name(&self) -> &str {
        match (self.negated, self.case_insensitive) {
            (false, false) => "LIKE",
            (true, false) => "NOT LIKE",
            (false, true) => "ILIKE",
            (true, true) => "NOT ILIKE",
        }
    }
}

impl fmt::Display for LikeExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} {} {}", self.expr, self.op_name(), self.pattern)
    }
}

impl PhysicalExpr for LikeExpr {
    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn nullable(&self, input_schema: &Schema) -> Result<bool> {
        Ok(self.expr.nullable(input_schema)? || self.pattern.nullable(input_schema)?)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let lhs = self.expr.evaluate(batch)?;
        if !self.case_insensitive
            && let ColumnarValue::Array(array) = &lhs
            && matches!(
                array.data_type(),
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
            )
            && let Some((data_type, regex)) = self.literal_regex()
            && array.data_type() == data_type
        {
            let matches = |value: &str| regex.is_match(value) != self.negated;
            let result = match data_type {
                DataType::Utf8 => {
                    BooleanArray::from_unary(array.as_string::<i32>(), matches)
                }
                DataType::LargeUtf8 => {
                    BooleanArray::from_unary(array.as_string::<i64>(), matches)
                }
                DataType::Utf8View => {
                    BooleanArray::from_unary(array.as_string_view(), matches)
                }
                _ => unreachable!(),
            };
            return Ok(ColumnarValue::Array(Arc::new(result)));
        }
        let rhs = self.pattern.evaluate(batch)?;
        match (self.negated, self.case_insensitive) {
            (false, false) => apply_cmp(Operator::LikeMatch, &lhs, &rhs),
            (false, true) => apply_cmp(Operator::ILikeMatch, &lhs, &rhs),
            (true, false) => apply_cmp(Operator::NotLikeMatch, &lhs, &rhs),
            (true, true) => apply_cmp(Operator::NotILikeMatch, &lhs, &rhs),
        }
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.expr, &self.pattern]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(Arc::new(LikeExpr::new(
            self.negated,
            self.case_insensitive,
            Arc::clone(&children[0]),
            Arc::clone(&children[1]),
        )))
    }

    fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.expr.fmt_sql(f)?;
        write!(f, " {} ", self.op_name())?;
        self.pattern.fmt_sql(f)
    }

    #[cfg(feature = "proto")]
    fn try_to_proto(
        &self,
        ctx: &datafusion_physical_expr_common::physical_expr::proto_encode::PhysicalExprEncodeCtx<'_>,
    ) -> Result<Option<datafusion_proto_models::protobuf::PhysicalExprNode>> {
        use datafusion_proto_models::protobuf;

        let Self {
            negated,
            case_insensitive,
            expr,
            pattern,
            ..
        } = self;

        Ok(Some(protobuf::PhysicalExprNode {
            expr_id: None,
            expr_type: Some(protobuf::physical_expr_node::ExprType::LikeExpr(Box::new(
                protobuf::PhysicalLikeExprNode {
                    negated: *negated,
                    case_insensitive: *case_insensitive,
                    expr: Some(Box::new(ctx.encode_child(expr)?)),
                    pattern: Some(Box::new(ctx.encode_child(pattern)?)),
                },
            ))),
        }))
    }
}

#[cfg(feature = "proto")]
impl LikeExpr {
    /// Reconstruct a [`LikeExpr`] from its protobuf representation.
    ///
    /// Takes the whole [`PhysicalExprNode`] so the decode signature matches
    /// other migrated expressions and can inspect outer-node metadata if
    /// needed in the future.
    ///
    /// [`PhysicalExprNode`]: datafusion_proto_models::protobuf::PhysicalExprNode
    pub fn try_from_proto(
        node: &datafusion_proto_models::protobuf::PhysicalExprNode,
        ctx: &datafusion_physical_expr_common::physical_expr::proto_decode::PhysicalExprDecodeCtx<'_>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        use datafusion_physical_expr_common::expect_expr_variant;
        use datafusion_proto_models::protobuf;

        let like_expr = expect_expr_variant!(
            node,
            protobuf::physical_expr_node::ExprType::LikeExpr,
            "LikeExpr",
        );

        let protobuf::PhysicalLikeExprNode {
            negated,
            case_insensitive,
            expr,
            pattern,
        } = like_expr.as_ref();

        Ok(Arc::new(LikeExpr::new(
            *negated,
            *case_insensitive,
            ctx.decode_required_expression(expr.as_deref(), "LikeExpr", "expr")?,
            ctx.decode_required_expression(pattern.as_deref(), "LikeExpr", "pattern")?,
        )))
    }
}

fn compile_like_regex(pattern: &str) -> Option<Regex> {
    let body = pattern.strip_prefix('%').unwrap_or(pattern);
    let body = body.strip_suffix('%').unwrap_or(body);
    // Leave SQL escapes and Arrow's equality, prefix, suffix, and substring kernels to Arrow.
    if pattern.contains('\\') || !body.contains(['%', '_']) {
        return None;
    }

    let mut source = String::new();
    if !pattern.starts_with('%') {
        source.push('^');
    }
    let mut start = 0;
    for (index, wildcard) in body.match_indices(['%', '_']) {
        source.push_str(&regex::escape(&body[start..index]));
        source.push_str(if wildcard == "%" { ".*" } else { "." });
        start = index + 1;
    }
    source.push_str(&regex::escape(&body[start..]));
    if !pattern.ends_with('%') {
        source.push('$');
    }
    // Arrow reports the usual pattern error if the builder rejects this regex.
    RegexBuilder::new(&source)
        .dot_matches_new_line(true)
        .build()
        .ok()
}

/// used for optimize Dictionary like
fn can_like_type(from_type: &DataType) -> bool {
    match from_type {
        DataType::Dictionary(_, inner_type_from) => **inner_type_from == DataType::Utf8,
        _ => false,
    }
}

/// Create a like expression, erroring if the argument types are not compatible.
pub fn like(
    negated: bool,
    case_insensitive: bool,
    expr: Arc<dyn PhysicalExpr>,
    pattern: Arc<dyn PhysicalExpr>,
    input_schema: &Schema,
) -> Result<Arc<dyn PhysicalExpr>> {
    let expr_type = &expr.data_type(input_schema)?;
    let pattern_type = &pattern.data_type(input_schema)?;
    assert_or_internal_err!(
        expr_type.eq(pattern_type) || can_like_type(expr_type),
        "The type of {expr_type} AND {pattern_type} of like physical should be same"
    );
    Ok(Arc::new(LikeExpr::new(
        negated,
        case_insensitive,
        expr,
        pattern,
    )))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::expressions::{col, lit};
    use arrow::array::*;
    use arrow::compute::cast;
    use arrow::datatypes::Field;
    use datafusion_common::cast::as_boolean_array;
    use datafusion_physical_expr_common::physical_expr::fmt_sql;

    macro_rules! test_like {
        ($A_VEC:expr, $B_VEC:expr, $VEC:expr, $NULLABLE: expr, $NEGATED:expr, $CASE_INSENSITIVE:expr,) => {{
            let schema = Schema::new(vec![
                Field::new("a", DataType::Utf8, $NULLABLE),
                Field::new("b", DataType::Utf8, $NULLABLE),
            ]);
            let a = StringArray::from($A_VEC);
            let b = StringArray::from($B_VEC);

            let expression = like(
                $NEGATED,
                $CASE_INSENSITIVE,
                col("a", &schema)?,
                col("b", &schema)?,
                &schema,
            )?;
            let batch = RecordBatch::try_new(
                Arc::new(schema.clone()),
                vec![Arc::new(a), Arc::new(b)],
            )?;

            // compute
            let result = expression
                .evaluate(&batch)?
                .into_array(batch.num_rows())
                .expect("Failed to convert to array");
            let result =
                as_boolean_array(&result).expect("failed to downcast to BooleanArray");
            let expected = &BooleanArray::from($VEC);
            assert_eq!(expected, result);
        }};
    }

    #[test]
    fn like_op() -> Result<()> {
        test_like!(
            vec!["hello world", "world"],
            vec!["%hello%", "%hello%"],
            vec![true, false],
            false,
            false,
            false,
        ); // like
        test_like!(
            vec![Some("hello world"), None, Some("world")],
            vec![Some("%hello%"), None, Some("%hello%")],
            vec![Some(false), None, Some(true)],
            true,
            true,
            false,
        ); // not like
        test_like!(
            vec!["hello world", "world"],
            vec!["%helLo%", "%helLo%"],
            vec![true, false],
            false,
            false,
            true,
        ); // ilike
        test_like!(
            vec![Some("hello world"), None, Some("world")],
            vec![Some("%helLo%"), None, Some("%helLo%")],
            vec![Some(false), None, Some(true)],
            true,
            true,
            true,
        ); // not ilike

        Ok(())
    }

    #[test]
    fn literal_like_matches_arrow_across_batches() -> Result<()> {
        let input = StringArray::from(vec![
            Some("p\n"),
            Some("p雪"),
            Some("pe\u{0301}"),
            Some("p"),
            Some("p.b\nz"),
            Some("p_z"),
            Some("abc.z\n"),
            Some(""),
            None,
        ]);
        for data_type in [DataType::Utf8, DataType::LargeUtf8, DataType::Utf8View] {
            let schema =
                Arc::new(Schema::new(vec![Field::new("s", data_type.clone(), true)]));
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![cast(&input, &data_type)?],
            )?;
            for (negated, op) in
                [(false, Operator::LikeMatch), (true, Operator::NotLikeMatch)]
            {
                for pattern in [
                    Some("p_"),
                    Some("p%z"),
                    Some("%.z_%"),
                    Some("p%%"),
                    Some("%p%"),
                    Some(r"p\_%"),
                    None,
                ] {
                    let literal = ScalarValue::Utf8(pattern.map(str::to_owned))
                        .cast_to(&data_type)?;
                    let expression = like(
                        negated,
                        false,
                        col("s", &schema)?,
                        lit(literal.clone()),
                        &schema,
                    )?;
                    for batch in [batch.slice(0, 0), batch.slice(0, 4), batch.slice(4, 5)]
                    {
                        let expected = apply_cmp(
                            op,
                            &ColumnarValue::Array(Arc::clone(batch.column(0))),
                            &ColumnarValue::Scalar(literal.clone()),
                        )?;
                        let actual =
                            expression.evaluate(&batch)?.into_array(batch.num_rows())?;
                        let expected = expected.into_array(batch.num_rows())?;
                        assert_eq!(
                            actual.as_ref(),
                            expected.as_ref(),
                            "{op:?} {pattern:?} on {data_type}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_fmt_sql() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Utf8, false),
            Field::new("b", DataType::Utf8, false),
        ]);

        let expr = like(
            false,
            false,
            col("a", &schema)?,
            col("b", &schema)?,
            &schema,
        )?;

        // Display format
        let display_string = expr.to_string();
        assert_eq!(display_string, "a@0 LIKE b@1");

        // fmt_sql format
        let sql_string = fmt_sql(expr.as_ref()).to_string();
        assert_eq!(sql_string, "a LIKE b");

        Ok(())
    }
}

/// Tests for the `try_to_proto` / `try_from_proto` hooks.
#[cfg(all(test, feature = "proto"))]
mod proto_tests {
    use super::*;
    use crate::expressions::{Column, col};
    use crate::proto_test_util::{
        StubDecoder, StubEncoder, UnreachableDecoder, column_node,
    };
    use arrow::datatypes::Field;
    use datafusion_common::DataFusionError;
    use datafusion_physical_expr_common::physical_expr::proto_decode::PhysicalExprDecodeCtx;
    use datafusion_physical_expr_common::physical_expr::proto_encode::PhysicalExprEncodeCtx;
    use datafusion_proto_models::protobuf::{
        PhysicalExprNode, PhysicalLikeExprNode, physical_expr_node,
    };

    /// Build a `LikeExpr` proto node with the given children.
    fn like_node(
        negated: bool,
        case_insensitive: bool,
        expr: Option<Box<PhysicalExprNode>>,
        pattern: Option<Box<PhysicalExprNode>>,
    ) -> PhysicalExprNode {
        PhysicalExprNode {
            expr_id: None,
            expr_type: Some(physical_expr_node::ExprType::LikeExpr(Box::new(
                PhysicalLikeExprNode {
                    negated,
                    case_insensitive,
                    expr,
                    pattern,
                },
            ))),
        }
    }

    /// A `LikeExpr` over two `Utf8` columns with both flags set, so the
    /// `negated` / `case_insensitive` wiring is actually exercised.
    fn like_fixture() -> LikeExpr {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Utf8, false),
            Field::new("b", DataType::Utf8, false),
        ]);
        LikeExpr::new(
            true,
            true,
            col("a", &schema).unwrap(),
            col("b", &schema).unwrap(),
        )
    }

    #[test]
    fn try_to_proto_encodes_like_expr() {
        let like = like_fixture();
        let encoder = StubEncoder::ok();
        let ctx = PhysicalExprEncodeCtx::new(&encoder);

        let node = like
            .try_to_proto(&ctx)
            .unwrap()
            .expect("LikeExpr should encode to Some(node)");

        // Built-in exprs never set expr_id; only dynamic filters do.
        assert!(node.expr_id.is_none());
        let like_node = match node.expr_type {
            Some(physical_expr_node::ExprType::LikeExpr(boxed)) => *boxed,
            other => panic!("expected a LikeExpr node, got {other:?}"),
        };
        assert!(like_node.negated);
        assert!(like_node.case_insensitive);
        assert!(like_node.expr.is_some());
        assert!(like_node.pattern.is_some());
    }

    #[test]
    fn try_to_proto_propagates_expr_encode_error() {
        let like = like_fixture();
        let encoder = StubEncoder::failing_on(1);
        let ctx = PhysicalExprEncodeCtx::new(&encoder);
        let err = like.try_to_proto(&ctx).unwrap_err();
        assert!(matches!(err, DataFusionError::Internal(msg) if msg.contains("call 1")));
    }

    #[test]
    fn try_to_proto_propagates_pattern_encode_error() {
        let like = like_fixture();
        let encoder = StubEncoder::failing_on(2);
        let ctx = PhysicalExprEncodeCtx::new(&encoder);
        let err = like.try_to_proto(&ctx).unwrap_err();
        assert!(matches!(err, DataFusionError::Internal(msg) if msg.contains("call 2")));
    }

    #[test]
    fn try_from_proto_decodes_like_expr() {
        let node = like_node(
            true,
            true,
            Some(Box::new(column_node("a"))),
            Some(Box::new(column_node("b"))),
        );
        let schema = Schema::empty();
        let decoder = StubDecoder::ok();
        let ctx = PhysicalExprDecodeCtx::new(&schema, &decoder);

        let decoded = LikeExpr::try_from_proto(&node, &ctx).unwrap();
        let like = decoded
            .downcast_ref::<LikeExpr>()
            .expect("decoded expr should be a LikeExpr");
        assert!(like.negated());
        assert!(like.case_insensitive());
        assert!(like.expr().downcast_ref::<Column>().is_some());
        assert!(like.pattern().downcast_ref::<Column>().is_some());
    }

    #[test]
    fn try_from_proto_rejects_non_like_node() {
        let node = column_node("a");
        let schema = Schema::empty();
        let decoder = UnreachableDecoder;
        let ctx = PhysicalExprDecodeCtx::new(&schema, &decoder);
        let err = LikeExpr::try_from_proto(&node, &ctx).unwrap_err();
        assert!(matches!(
            err,
            DataFusionError::Internal(msg) if msg.contains("PhysicalExprNode is not a LikeExpr")
        ));
    }

    #[test]
    fn try_from_proto_rejects_missing_expr() {
        let node = like_node(false, false, None, Some(Box::new(column_node("b"))));
        let schema = Schema::empty();
        let decoder = UnreachableDecoder;
        let ctx = PhysicalExprDecodeCtx::new(&schema, &decoder);
        let err = LikeExpr::try_from_proto(&node, &ctx).unwrap_err();
        assert!(matches!(
            err,
            DataFusionError::Internal(msg) if msg.contains("LikeExpr is missing required field 'expr'")
        ));
    }

    #[test]
    fn try_from_proto_rejects_missing_pattern() {
        let node = like_node(false, false, Some(Box::new(column_node("a"))), None);
        let schema = Schema::empty();
        // `expr` is present, so it is decoded before the missing-`pattern`
        // check fires; use a decoder that succeeds for that first child.
        let decoder = StubDecoder::ok();
        let ctx = PhysicalExprDecodeCtx::new(&schema, &decoder);
        let err = LikeExpr::try_from_proto(&node, &ctx).unwrap_err();
        assert!(matches!(
            err,
            DataFusionError::Internal(msg) if msg.contains("LikeExpr is missing required field 'pattern'")
        ));
    }

    #[test]
    fn try_from_proto_propagates_expr_decode_error() {
        let node = like_node(
            false,
            false,
            Some(Box::new(column_node("a"))),
            Some(Box::new(column_node("b"))),
        );
        let schema = Schema::empty();
        let decoder = StubDecoder::failing_on(1);
        let ctx = PhysicalExprDecodeCtx::new(&schema, &decoder);
        let err = LikeExpr::try_from_proto(&node, &ctx).unwrap_err();
        assert!(matches!(err, DataFusionError::Internal(msg) if msg.contains("call 1")));
    }

    #[test]
    fn try_from_proto_propagates_pattern_decode_error() {
        let node = like_node(
            false,
            false,
            Some(Box::new(column_node("a"))),
            Some(Box::new(column_node("b"))),
        );
        let schema = Schema::empty();
        let decoder = StubDecoder::failing_on(2);
        let ctx = PhysicalExprDecodeCtx::new(&schema, &decoder);
        let err = LikeExpr::try_from_proto(&node, &ctx).unwrap_err();
        assert!(matches!(err, DataFusionError::Internal(msg) if msg.contains("call 2")));
    }
}
