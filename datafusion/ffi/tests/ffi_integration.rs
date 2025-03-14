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

/// Add an additional module here for convenience to scope this to only
/// when the feature integtation-tests is built
#[cfg(feature = "integration-tests")]
mod tests {

    use abi_stable::library::RootModule;
    use datafusion::common::record_batch;
    use datafusion::error::{DataFusionError, Result};
    use datafusion::logical_expr::ScalarUDF;
    use datafusion::prelude::{col, SessionContext};
    use datafusion_ffi::table_provider::ForeignTableProvider;
    use datafusion_ffi::tests::{create_record_batch, ForeignLibraryModuleRef};
    use datafusion_ffi::udf::ForeignScalarUDF;
    use std::path::Path;
    use std::sync::Arc;

    /// Compute the path to the library. It would be preferable to simply use
    /// abi_stable::library::development_utils::compute_library_path however
    /// our current CI pipeline has a `ci` profile that we need to use to
    /// find the library.
    pub fn compute_library_path<M: RootModule>(
        target_path: &Path,
    ) -> std::io::Result<std::path::PathBuf> {
        let all_paths = [
            "debug",
            "release",
            "ci",
            "llvm-cov-target/debug",
            "llvm-cov-target/ci",
        ]
        .iter()
        .map(|build_type| {
            let dir = target_path.join(build_type);
            let path = M::get_library_path(&dir.join("deps"));
            (dir, path)
        })
        .collect::<Vec<_>>();

        let best_path = all_paths
            .into_iter()
            .filter(|(_, path)| path.exists())
            .filter_map(|(dir, path)| path.metadata().map(|m| (dir, m)).ok())
            .filter_map(|(dir, meta)| meta.modified().map(|m| (dir, m)).ok())
            .max_by_key(|(_, date)| *date)
            .map(|(dir, _)| dir)
            .unwrap_or_else(|| target_path.join("debug"));

        Ok(best_path)
    }

    fn get_module() -> Result<ForeignLibraryModuleRef> {
        let expected_version = datafusion_ffi::version();

        let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let target_dir = crate_root
            .parent()
            .expect("Failed to find crate parent")
            .parent()
            .expect("Failed to find workspace root")
            .join("target");

        // Find the location of the library. This is specific to the build environment,
        // so you will need to change the approach here based on your use case.
        // let target: &std::path::Path = "../../../../target/".as_ref();
        let library_path =
            compute_library_path::<ForeignLibraryModuleRef>(target_dir.as_path())
                .map_err(|e| DataFusionError::External(Box::new(e)))?
                .join("deps");

        // Load the module
        let module = ForeignLibraryModuleRef::load_from_directory(&library_path)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        assert_eq!(
            module
                .version()
                .expect("Unable to call version on FFI module")(),
            expected_version
        );

        Ok(module)
    }

    /// It is important that this test is in the `tests` directory and not in the
    /// library directory so we can verify we are building a dynamic library and
    /// testing it via a different executable.
    async fn test_table_provider(synchronous: bool) -> Result<()> {
        let table_provider_module = get_module()?;

        // By calling the code below, the table provided will be created within
        // the module's code.
        let ffi_table_provider = table_provider_module.create_table().ok_or(
            DataFusionError::NotImplemented(
                "External table provider failed to implement create_table".to_string(),
            ),
        )?(synchronous);

        // In order to access the table provider within this executable, we need to
        // turn it into a `ForeignTableProvider`.
        let foreign_table_provider: ForeignTableProvider = (&ffi_table_provider).into();

        let ctx = SessionContext::new();

        // Display the data to show the full cycle works.
        ctx.register_table("external_table", Arc::new(foreign_table_provider))?;
        let df = ctx.table("external_table").await?;
        let results = df.collect().await?;

        assert_eq!(results.len(), 3);
        assert_eq!(results[0], create_record_batch(1, 5));
        assert_eq!(results[1], create_record_batch(6, 1));
        assert_eq!(results[2], create_record_batch(7, 5));

        Ok(())
    }

    #[tokio::test]
    async fn async_test_table_provider() -> Result<()> {
        test_table_provider(false).await
    }

    #[tokio::test]
    async fn sync_test_table_provider() -> Result<()> {
        test_table_provider(true).await
    }

    /// This test validates that we can load an external module and use a scalar
    /// udf defined in it via the foreign function interface. In this case we are
    /// using the abs() function as our scalar UDF.
    #[tokio::test]
    async fn test_scalar_udf() -> Result<()> {
        let module = get_module()?;

        let ffi_abs_func =
            module
                .create_scalar_udf()
                .ok_or(DataFusionError::NotImplemented(
                    "External table provider failed to implement create_scalar_udf"
                        .to_string(),
                ))?();
        let foreign_abs_func: ForeignScalarUDF = (&ffi_abs_func).try_into()?;

        let udf: ScalarUDF = foreign_abs_func.into();

        let ctx = SessionContext::default();
        let df = ctx.read_batch(create_record_batch(-5, 5))?;

        let df = df
            .with_column("abs_a", udf.call(vec![col("a")]))?
            .with_column("abs_b", udf.call(vec![col("b")]))?;

        let result = df.collect().await?;

        let expected = record_batch!(
            ("a", Int32, vec![-5, -4, -3, -2, -1]),
            ("b", Float64, vec![-5., -4., -3., -2., -1.]),
            ("abs_a", Int32, vec![5, 4, 3, 2, 1]),
            ("abs_b", Float64, vec![5., 4., 3., 2., 1.])
        )?;

        assert!(result.len() == 1);
        assert!(result[0] == expected);

        Ok(())
    }
}
