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

extern crate arrow;
#[macro_use]
extern crate criterion;

use crate::criterion::Criterion;
use criterion::{AxisScale, BenchmarkId, PlotConfiguration, Throughput};
use datafusion_common::tree_node::{
    DynTreeNode, RecursiveNode, TreeNode, TreeNodeRecursion, TreeNodeVisitor,
};
use std::fmt::format;
use std::sync::Arc;

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct RecursiveTree<T> {
    pub(crate) data: Arc<T>,
    pub(crate) children: Vec<Arc<Self>>,
}

impl<T> DynTreeNode for RecursiveTree<T> {
    fn arc_children(&self) -> Vec<&Arc<Self>> {
        self.children.iter().collect()
    }

    fn with_new_arc_children(
        self: Arc<Self>,
        new_children: Vec<Arc<Self>>,
    ) -> datafusion_common::Result<Arc<Self>> {
        Ok(Arc::new(RecursiveTree {
            data: self.data.clone(),
            children: new_children,
        }))
    }
}

fn make_tree(width: usize, height: usize) -> Arc<RecursiveTree<String>> {
    let mut node = Arc::new(RecursiveTree {
        data: Arc::new("n".to_string()),
        children: vec![],
    });

    for _ in 0..height {
        let children = std::iter::repeat(&node)
            .take(width)
            .cloned()
            .collect::<Vec<_>>();
        node = node.with_new_arc_children(children).unwrap();
    }

    node
}

struct Visitor {
    n: usize,
}

impl Visitor {
    fn new(n: usize) -> Self {
        Self { n }
    }
}
impl<'n> TreeNodeVisitor<'n> for Visitor {
    type Node = Arc<RecursiveTree<String>>;

    fn f_down(
        &mut self,
        _node: &'n Self::Node,
    ) -> datafusion_common::Result<TreeNodeRecursion> {
        self.n += 1;
        Ok(TreeNodeRecursion::Continue)
    }

    fn f_up(
        &mut self,
        _node: &'n Self::Node,
    ) -> datafusion_common::Result<TreeNodeRecursion> {
        self.n += 1;
        Ok(TreeNodeRecursion::Continue)
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let width = 10;

    let mut group = c.benchmark_group(format!("Visit Tree width={:}", width));
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    group.plot_config(plot_config);

    for height in [2, 3, 4, 5, 6].iter() {
        group.bench_with_input(BenchmarkId::new("Recursive", height), height, |b, h| {
            let mut visitor = Visitor::new(0);
            let tree = make_tree(width, *h);

            b.iter(|| tree.visit(&mut visitor))
        });

        group.bench_with_input(BenchmarkId::new("Iterative", height), height, |b, h| {
            let mut visitor = Visitor::new(0);
            let tree = make_tree(width, *h);
            b.iter(|| tree.visit_iterative(&mut visitor))
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
