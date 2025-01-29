// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Defines the FIRST_VALUE/LAST_VALUE aggregations.

use std::any::Any;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::mem::size_of_val;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, BooleanArray};
use arrow::compute::{
    self, lexsort_to_indices, take_arrays, LexicographicalComparator, SortColumn,
};
use arrow::datatypes::{DataType, Field};
use arrow::row::{RowConverter, SortField};
use datafusion_common::utils::{compare_rows, get_row_at_idx};
use datafusion_common::{
    arrow_datafusion_err, internal_err, DataFusionError, Result, ScalarValue,
};
use datafusion_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion_expr::utils::{format_state_name, AggregateOrderSensitivity};
use datafusion_expr::{
    Accumulator, AggregateUDFImpl, Documentation, Expr, ExprFunctionExt, Signature,
    SortExpr, Volatility,
};
use datafusion_functions_aggregate_common::utils::get_sort_options;
use datafusion_macros::user_doc;
use datafusion_physical_expr_common::sort_expr::LexOrdering;

create_func!(FirstValue, first_value_udaf);

/// Returns the first value in a group of values.
pub fn first_value(expression: Expr, order_by: Option<Vec<SortExpr>>) -> Expr {
    if let Some(order_by) = order_by {
        first_value_udaf()
            .call(vec![expression])
            .order_by(order_by)
            .build()
            // guaranteed to be `Expr::AggregateFunction`
            .unwrap()
    } else {
        first_value_udaf().call(vec![expression])
    }
}

#[user_doc(
    doc_section(label = "General Functions"),
    description = "Returns the first element in an aggregation group according to the requested ordering. If no ordering is given, returns an arbitrary element from the group.",
    syntax_example = "first_value(expression [ORDER BY expression])",
    sql_example = r#"```sql
> SELECT first_value(column_name ORDER BY other_column) FROM table_name;
+-----------------------------------------------+
| first_value(column_name ORDER BY other_column)|
+-----------------------------------------------+
| first_element                                 |
+-----------------------------------------------+
```"#,
    standard_argument(name = "expression",)
)]
pub struct FirstValue {
    signature: Signature,
    requirement_satisfied: bool,
}

impl Debug for FirstValue {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("FirstValue")
            .field("name", &self.name())
            .field("signature", &self.signature)
            .field("accumulator", &"<FUNC>")
            .finish()
    }
}

impl Default for FirstValue {
    fn default() -> Self {
        Self::new()
    }
}

impl FirstValue {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
            requirement_satisfied: false,
        }
    }

    fn with_requirement_satisfied(mut self, requirement_satisfied: bool) -> Self {
        self.requirement_satisfied = requirement_satisfied;
        self
    }
}

impl AggregateUDFImpl for FirstValue {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "first_value"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(arg_types[0].clone())
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        let ordering_dtypes = acc_args
            .ordering_req
            .iter()
            .map(|e| e.expr.data_type(acc_args.schema))
            .collect::<Result<Vec<_>>>()?;

        // When requirement is empty, or it is signalled by outside caller that
        // the ordering requirement is/will be satisfied.
        let requirement_satisfied =
            acc_args.ordering_req.is_empty() || self.requirement_satisfied;

        FirstLastAccumulator::try_new(
            acc_args.return_type,
            &ordering_dtypes,
            acc_args.ordering_req.clone(),
            acc_args.ignore_nulls,
            Ordering::Less,
        )
        .map(|acc| Box::new(acc.with_requirement_satisfied(requirement_satisfied)) as _)
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<Field>> {
        let mut fields = vec![Field::new(
            format_state_name(args.name, "first_value"),
            args.return_type.clone(),
            true,
        )];
        fields.extend(args.ordering_fields.to_vec());
        fields.push(Field::new("is_set", DataType::Boolean, true));
        Ok(fields)
    }

    fn aliases(&self) -> &[String] {
        &[]
    }

    fn with_beneficial_ordering(
        self: Arc<Self>,
        beneficial_ordering: bool,
    ) -> Result<Option<Arc<dyn AggregateUDFImpl>>> {
        Ok(Some(Arc::new(
            FirstValue::new().with_requirement_satisfied(beneficial_ordering),
        )))
    }

    fn order_sensitivity(&self) -> AggregateOrderSensitivity {
        AggregateOrderSensitivity::Beneficial
    }

    fn reverse_expr(&self) -> datafusion_expr::ReversedUDAF {
        datafusion_expr::ReversedUDAF::Reversed(last_value_udaf())
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

make_udaf_expr_and_func!(
    LastValue,
    last_value,
    "Returns the last value in a group of values.",
    last_value_udaf
);

#[user_doc(
    doc_section(label = "General Functions"),
    description = "Returns the last element in an aggregation group according to the requested ordering. If no ordering is given, returns an arbitrary element from the group.",
    syntax_example = "last_value(expression [ORDER BY expression])",
    sql_example = r#"```sql
> SELECT last_value(column_name ORDER BY other_column) FROM table_name;
+-----------------------------------------------+
| last_value(column_name ORDER BY other_column) |
+-----------------------------------------------+
| last_element                                  |
+-----------------------------------------------+
```"#,
    standard_argument(name = "expression",)
)]
pub struct LastValue {
    signature: Signature,
    requirement_satisfied: bool,
}

impl Debug for LastValue {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("LastValue")
            .field("name", &self.name())
            .field("signature", &self.signature)
            .field("accumulator", &"<FUNC>")
            .finish()
    }
}

impl Default for LastValue {
    fn default() -> Self {
        Self::new()
    }
}

impl LastValue {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
            requirement_satisfied: false,
        }
    }

    fn with_requirement_satisfied(mut self, requirement_satisfied: bool) -> Self {
        self.requirement_satisfied = requirement_satisfied;
        self
    }
}

impl AggregateUDFImpl for LastValue {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "last_value"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(arg_types[0].clone())
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        let ordering_dtypes = acc_args
            .ordering_req
            .iter()
            .map(|e| e.expr.data_type(acc_args.schema))
            .collect::<Result<Vec<_>>>()?;

        let requirement_satisfied =
            acc_args.ordering_req.is_empty() || self.requirement_satisfied;

        FirstLastAccumulator::try_new(
            acc_args.return_type,
            &ordering_dtypes,
            acc_args.ordering_req.clone(),
            acc_args.ignore_nulls,
            Ordering::Greater,
        )
        .map(|acc| Box::new(acc.with_requirement_satisfied(requirement_satisfied)) as _)
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<Field>> {
        let StateFieldsArgs {
            name,
            input_types,
            return_type: _,
            ordering_fields,
            is_distinct: _,
        } = args;
        let mut fields = vec![Field::new(
            format_state_name(name, "last_value"),
            input_types[0].clone(),
            true,
        )];
        fields.extend(ordering_fields.to_vec());
        fields.push(Field::new("is_set", DataType::Boolean, true));
        Ok(fields)
    }

    fn aliases(&self) -> &[String] {
        &[]
    }

    fn with_beneficial_ordering(
        self: Arc<Self>,
        beneficial_ordering: bool,
    ) -> Result<Option<Arc<dyn AggregateUDFImpl>>> {
        Ok(Some(Arc::new(
            LastValue::new().with_requirement_satisfied(beneficial_ordering),
        )))
    }

    fn order_sensitivity(&self) -> AggregateOrderSensitivity {
        AggregateOrderSensitivity::Beneficial
    }

    fn reverse_expr(&self) -> datafusion_expr::ReversedUDAF {
        datafusion_expr::ReversedUDAF::Reversed(first_value_udaf())
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

#[derive(Debug)]
struct FirstLastAccumulator {
    value: ScalarValue,

    // At the beginning, `is_set` is false, which means `first` is not seen yet.
    // Once we see the first value, we set the `is_set` flag and do not update `first` anymore.
    is_set: bool, // todo: kill is_set

    // Stores whether incoming data already satisfies the ordering requirement.
    requirement_satisfied: bool,

    // Stores ordering values, of the aggregator requirement corresponding to first value
    // of the aggregator. These values are used during merging of multiple partitions.
    orderings: Vec<ScalarValue>,
    // Stores the applicable ordering requirement.
    ordering_req: LexOrdering,
    // Ignore null values.
    ignore_nulls: bool,

    target: Ordering,

    potentially_sortfields: Vec<SortField>,
}

impl FirstLastAccumulator {
    pub fn try_new(
        data_type: &DataType,
        ordering_dtypes: &[DataType],
        ordering_req: LexOrdering,
        ignore_nulls: bool,
        target: Ordering,
    ) -> Result<Self> {
        let orderings = ordering_dtypes
            .iter()
            .map(ScalarValue::try_from)
            .collect::<Result<Vec<_>>>()?;
        let requirement_satisfied = ordering_req.is_empty();

        let fields: Vec<_> = ordering_dtypes
            .iter()
            .map(|a| SortField::new(a.clone()))
            .collect();

        ScalarValue::try_from(data_type).map(|value| Self {
            value,
            is_set: false,
            orderings,
            ordering_req,
            requirement_satisfied,
            ignore_nulls,
            potentially_sortfields: fields,
            target,
        })
    }

    pub fn with_requirement_satisfied(mut self, requirement_satisfied: bool) -> Self {
        self.requirement_satisfied = requirement_satisfied;
        self
    }

    // Updates state with the values in the given row.
    fn update_with_new_row(&mut self, row: &[ScalarValue]) {
        self.value = row[0].clone();
        self.orderings = row[1..].to_vec();
        self.is_set = true;
    }

    fn get_first_idx(&self, values: &[ArrayRef]) -> Result<Option<usize>> {
        // todo: it's enough to do max(current, MAX(buffer))!
        let [value, ordering_values @ ..] = values else {
            return internal_err!("Empty row in FIRST_VALUE");
        };
        if self.requirement_satisfied {
            // Get first entry according to the pre-existing ordering (0th index):
            if self.ignore_nulls {
                // If ignoring nulls, find the first non-null value.
                for i in 0..value.len() {
                    if !value.is_null(i) {
                        return Ok(Some(i));
                    }
                }
                return Ok(None);
            } else {
                // If not ignoring nulls, return the first value if it exists.
                return Ok((!value.is_empty()).then_some(0));
            }
        }
        let sort_columns = ordering_values
            .iter()
            .zip(self.ordering_req.iter())
            .map(|(values, req)| SortColumn {
                values: Arc::clone(values),
                options: Some(req.options),
            })
            .collect::<Vec<_>>();

        // todo: i think we do store values per row. can do better! also. it SORTS items, while we need just top one!

        let comparator = LexicographicalComparator::try_new(&sort_columns)?;
        let mut found_value: Option<usize> = None;

        for index in 0..value.len() {
            // todo - rewrite to (0..n).max_by_key()!
            if self.ignore_nulls && value.is_null(index) {
                continue;
            }

            if let Some(current_found_value) = found_value {
                if comparator.compare(current_found_value, index) == self.target {
                    found_value = Some(index);
                }
            } else {
                found_value = Some(index);
            }
        }

        Ok(found_value)
    }

    fn get_first_idx_converter(&self, values: &[ArrayRef]) -> Result<Option<usize>> {
        let [value, ordering_values @ ..] = values else {
            return internal_err!("Empty row in FIRST_VALUE");
        };

        if self.requirement_satisfied {
            // Get first entry according to the pre-existing ordering (0th index):
            if self.ignore_nulls {
                // If ignoring nulls, find the first non-null value.
                for i in 0..value.len() {
                    if !value.is_null(i) {
                        return Ok(Some(i));
                    }
                }
                return Ok(None);
            } else {
                // If not ignoring nulls, return the first value if it exists.
                return Ok((!value.is_empty()).then_some(0));
            }
        }

        let converter = RowConverter::new(self.potentially_sortfields.clone())?;
        let rows = converter.convert_columns(ordering_values)?;

        debug_assert_eq!(value.len(), rows.num_rows());

        Ok((0..value.len()).min_by_key(|i| rows.row(*i)))
    }
}

impl Accumulator for FirstLastAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut result = vec![self.value.clone()];
        result.extend(self.orderings.iter().cloned());
        result.push(ScalarValue::Boolean(Some(self.is_set)));
        Ok(result)
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if !self.is_set {
            if let Some(first_idx) = self.get_first_idx_converter(values)? {
                let row = get_row_at_idx(values, first_idx)?;
                self.update_with_new_row(&row);
            }
        } else if !self.requirement_satisfied {
            if let Some(first_idx) = self.get_first_idx_converter(values)? {
                let row = get_row_at_idx(values, first_idx)?;
                let orderings = &row[1..];
                if compare_rows(
                    &self.orderings,
                    orderings,
                    &get_sort_options(self.ordering_req.as_ref()),
                )?
                .is_gt()
                {
                    self.update_with_new_row(&row);
                }
            }
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        // FIRST_VALUE(first1, first2, first3, ...)
        // last index contains is_set flag.
        let is_set_idx = states.len() - 1;
        let flags = states[is_set_idx].as_boolean();
        let filtered_states =
            filter_states_according_to_is_set(&states[0..is_set_idx], flags)?;
        // 1..is_set_idx range corresponds to ordering section
        let sort_columns = convert_to_sort_cols(
            &filtered_states[1..is_set_idx],
            self.ordering_req.as_ref(),
        );

        let comparator = LexicographicalComparator::try_new(&sort_columns)?;
        let mut min: Option<usize> = None;

        for n in 0..filtered_states[0].len() {
            if let Some(current_min) = min {
                if comparator.compare(current_min, n) == self.target {
                    min = Some(n);
                }
            } else {
                min = Some(n);
            }
        }

        if let Some(min) = min {
            let best_row = get_row_at_idx(&filtered_states, min)?;

            let best_ordering = &best_row[1..is_set_idx];
            let sort_options = get_sort_options(self.ordering_req.as_ref());

            if !self.is_set
                || compare_rows(&self.orderings, best_ordering, &sort_options)?
                    == self.target
            {
                self.update_with_new_row(&best_row[0..is_set_idx]);
            }
        }

        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        Ok(self.value.clone())
    }

    fn size(&self) -> usize {
        size_of_val(self) - size_of_val(&self.value)
            + self.value.size()
            + ScalarValue::size_of_vec(&self.orderings)
            - size_of_val(&self.orderings)
    }
}

/// Filters states according to the `is_set` flag at the last column and returns
/// the resulting states.
fn filter_states_according_to_is_set(
    states: &[ArrayRef],
    flags: &BooleanArray,
) -> Result<Vec<ArrayRef>> {
    states
        .iter()
        .map(|state| compute::filter(state, flags).map_err(|e| arrow_datafusion_err!(e)))
        .collect::<Result<Vec<_>>>()
}

/// Combines array refs and their corresponding orderings to construct `SortColumn`s.
fn convert_to_sort_cols(arrs: &[ArrayRef], sort_exprs: &LexOrdering) -> Vec<SortColumn> {
    arrs.iter()
        .zip(sort_exprs.iter())
        .map(|(item, sort_expr)| SortColumn {
            values: Arc::clone(item),
            options: Some(sort_expr.options),
        })
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use arrow::array::Int64Array;

    use super::*;

    #[test]
    fn test_first_last_value_value() -> Result<()> {
        let mut first_accumulator = FirstLastAccumulator::try_new(
            &DataType::Int64,
            &[],
            LexOrdering::default(),
            false,
            Ordering::Greater,
        )?;
        let mut last_accumulator = FirstLastAccumulator::try_new(
            &DataType::Int64,
            &[],
            LexOrdering::default(),
            false,
            Ordering::Less,
        )?;
        // first value in the tuple is start of the range (inclusive),
        // second value in the tuple is end of the range (exclusive)
        let ranges: Vec<(i64, i64)> = vec![(0, 10), (1, 11), (2, 13)];
        // create 3 ArrayRefs between each interval e.g from 0 to 9, 1 to 10, 2 to 12
        let arrs = ranges
            .into_iter()
            .map(|(start, end)| {
                Arc::new(Int64Array::from((start..end).collect::<Vec<_>>())) as ArrayRef
            })
            .collect::<Vec<_>>();
        for arr in arrs {
            // Once first_value is set, accumulator should remember it.
            // It shouldn't update first_value for each new batch
            first_accumulator.update_batch(&[Arc::clone(&arr)])?;
            // last_value should be updated for each new batch.
            last_accumulator.update_batch(&[arr])?;
        }
        // First Value comes from the first value of the first batch which is 0
        assert_eq!(first_accumulator.evaluate()?, ScalarValue::Int64(Some(0)));
        // Last value comes from the last value of the last batch which is 12
        assert_eq!(last_accumulator.evaluate()?, ScalarValue::Int64(Some(12)));
        Ok(())
    }

    #[test]
    fn test_first_last_state_after_merge() -> Result<()> {
        let ranges: Vec<(i64, i64)> = vec![(0, 10), (1, 11), (2, 13)];
        // create 3 ArrayRefs between each interval e.g from 0 to 9, 1 to 10, 2 to 12
        let arrs = ranges
            .into_iter()
            .map(|(start, end)| {
                Arc::new((start..end).collect::<Int64Array>()) as ArrayRef
            })
            .collect::<Vec<_>>();

        // FirstValueAccumulator
        let mut first_accumulator = FirstValueAccumulator::try_new(
            &DataType::Int64,
            &[],
            LexOrdering::default(),
            false,
        )?;

        first_accumulator.update_batch(&[Arc::clone(&arrs[0])])?;
        let state1 = first_accumulator.state()?;

        let mut first_accumulator = FirstValueAccumulator::try_new(
            &DataType::Int64,
            &[],
            LexOrdering::default(),
            false,
        )?;
        first_accumulator.update_batch(&[Arc::clone(&arrs[1])])?;
        let state2 = first_accumulator.state()?;

        assert_eq!(state1.len(), state2.len());

        let mut states = vec![];

        for idx in 0..state1.len() {
            states.push(compute::concat(&[
                &state1[idx].to_array()?,
                &state2[idx].to_array()?,
            ])?);
        }

        let mut first_accumulator = FirstValueAccumulator::try_new(
            &DataType::Int64,
            &[],
            LexOrdering::default(),
            false,
        )?;
        first_accumulator.merge_batch(&states)?;

        let merged_state = first_accumulator.state()?;
        assert_eq!(merged_state.len(), state1.len());

        // LastValueAccumulator
        let mut last_accumulator = LastValueAccumulator::try_new(
            &DataType::Int64,
            &[],
            LexOrdering::default(),
            false,
        )?;

        last_accumulator.update_batch(&[Arc::clone(&arrs[0])])?;
        let state1 = last_accumulator.state()?;

        let mut last_accumulator = LastValueAccumulator::try_new(
            &DataType::Int64,
            &[],
            LexOrdering::default(),
            false,
        )?;
        last_accumulator.update_batch(&[Arc::clone(&arrs[1])])?;
        let state2 = last_accumulator.state()?;

        assert_eq!(state1.len(), state2.len());

        let mut states = vec![];

        for idx in 0..state1.len() {
            states.push(compute::concat(&[
                &state1[idx].to_array()?,
                &state2[idx].to_array()?,
            ])?);
        }

        let mut last_accumulator = LastValueAccumulator::try_new(
            &DataType::Int64,
            &[],
            LexOrdering::default(),
            false,
        )?;
        last_accumulator.merge_batch(&states)?;

        let merged_state = last_accumulator.state()?;
        assert_eq!(merged_state.len(), state1.len());

        Ok(())
    }
}
