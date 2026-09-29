use crate::reference::{Scalar as _, close, product, transpose};

fn n(real: f64, imag: f64) -> Number {
    Number::new(real, imag)
}
fn ok(status: SparseStatus) {
    assert_eq!(status, SparseStatus::SUCCESS);
}

// Own heap buffers alongside their handle. Buffers never move or resize after creation.
struct Csr {
    raw: SparseMatrixHandle,
    _offsets: Vec<Index>,
    _indices: Vec<Index>,
    _values: Vec<Number>,
}
impl Drop for Csr {
    fn drop(&mut self) {
        let status = unsafe { mkl_sparse_destroy(self.raw) };
        if !std::thread::panicking() {
            ok(status);
        }
    }
}
fn csr_matrix(matrix: &[Number], rows: usize, cols: usize) -> Csr {
    assert_eq!(matrix.len(), rows * cols);
    let mut offsets = vec![0];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in 0..rows {
        for col in 0..cols {
            let value = matrix[row * cols + col];
            if value.parts() != (0., 0.) {
                indices.push(col as Index);
                values.push(value);
            }
        }
        offsets.push(indices.len() as Index);
    }
    let mut raw = ptr::null_mut();
    unsafe {
        ok(create_csr(
            &mut raw,
            SparseIndexBase::Zero,
            rows as Index,
            cols as Index,
            offsets.as_mut_ptr(),
            offsets.as_mut_ptr().add(1),
            indices.as_mut_ptr(),
            values.as_mut_ptr(),
        ));
    }
    Csr {
        raw,
        _offsets: offsets,
        _indices: indices,
        _values: values,
    }
}

fn exported_dense(raw: SparseMatrixHandle) -> Vec<Number> {
    let mut base = SparseIndexBase::Zero;
    let (mut rows, mut cols) = (0, 0);
    let (mut start, mut end, mut index, mut value) = (
        ptr::null_mut(),
        ptr::null_mut(),
        ptr::null_mut(),
        ptr::null_mut(),
    );
    unsafe {
        ok(export_csr(
            raw, &mut base, &mut rows, &mut cols, &mut start, &mut end, &mut index, &mut value,
        ));
        let shift = base as Index;
        let mut dense = vec![n(0., 0.); rows as usize * cols as usize];
        for row in 0..rows as usize {
            for position in (*start.add(row) - shift) as usize..(*end.add(row) - shift) as usize {
                let col = (*index.add(position) - shift) as usize;
                assert!(col < cols as usize);
                dense[row * cols as usize + col] =
                    dense[row * cols as usize + col].add(*value.add(position));
            }
        }
        dense
    }
}

#[test]
fn csr_one_based_empty_rows_and_export() {
    let mut offsets: [Index; 4] = [1, 3, 3, 4];
    let mut indices: [Index; 3] = [1, 3, 2];
    let mut values = [n(2., 1.), n(-1., 2.), n(3., -1.)];
    let mut raw = ptr::null_mut();
    unsafe {
        ok(create_csr(
            &mut raw,
            SparseIndexBase::One,
            3,
            3,
            offsets.as_mut_ptr(),
            offsets.as_mut_ptr().add(1),
            indices.as_mut_ptr(),
            values.as_mut_ptr(),
        ));
    }
    let handle = Handle(raw, PhantomData);
    let matrix = [
        values[0],
        n(0., 0.),
        values[1],
        n(0., 0.),
        n(0., 0.),
        n(0., 0.),
        n(0., 0.),
        values[2],
        n(0., 0.),
    ];
    close(&exported_dense(handle.0), &matrix);
    let x = [n(1., 2.), n(3., -1.), n(-2., 1.)];
    let original = [n(2., -1.); 3];
    let alpha = n(2., 1.);
    let beta = n(-1., 2.);
    for operation in [
        OP,
        SparseOperation::Transpose,
        SparseOperation::ConjugateTranspose,
    ] {
        let modified = if operation == OP {
            matrix.to_vec()
        } else {
            transpose(
                &matrix,
                3,
                3,
                operation == SparseOperation::ConjugateTranspose,
            )
        };
        let expected: Vec<_> = product(&modified, &x, 3, 3, 1)
            .iter()
            .zip(original)
            .map(|(&v, y)| alpha.mul(v).add(beta.mul(y)))
            .collect();
        let mut y = original;
        unsafe {
            ok(mv(
                operation,
                alpha,
                handle.0,
                GENERAL,
                x.as_ptr(),
                beta,
                y.as_mut_ptr(),
            ));
        }
        close(&y, &expected);
    }
}

#[test]
fn csc_creation_export_and_conversion() {
    let mut offsets: [Index; 4] = [0, 1, 3, 4];
    let mut rows: [Index; 4] = [1, 0, 2, 1];
    let mut values = [n(2., 1.), n(3., -1.), n(-1., 2.), n(4., 1.)];
    let mut raw = ptr::null_mut();
    unsafe {
        ok(create_csc(
            &mut raw,
            SparseIndexBase::Zero,
            3,
            3,
            offsets.as_mut_ptr(),
            offsets.as_mut_ptr().add(1),
            rows.as_mut_ptr(),
            values.as_mut_ptr(),
        ));
    }
    let handle = Handle(raw, PhantomData);
    let (mut base, mut nr, mut nc) = (SparseIndexBase::One, 0, 0);
    let (mut start, mut end, mut index, mut value) = (
        ptr::null_mut(),
        ptr::null_mut(),
        ptr::null_mut(),
        ptr::null_mut(),
    );
    unsafe {
        ok(export_csc(
            handle.0, &mut base, &mut nr, &mut nc, &mut start, &mut end, &mut index, &mut value,
        ));
        assert_eq!((base, nr, nc), (SparseIndexBase::Zero, 3, 3));
        assert_eq!(std::slice::from_raw_parts(start, 3), &[0, 1, 3]);
        assert_eq!(std::slice::from_raw_parts(end, 3), &[1, 3, 4]);
        assert_eq!(std::slice::from_raw_parts(index, 4), &rows);
        close(std::slice::from_raw_parts(value, 4), &values);
    }
    let mut converted = ptr::null_mut();
    unsafe {
        ok(mkl_sparse_convert_csr(handle.0, OP, &mut converted));
    }
    let converted = Handle(converted, PhantomData);
    close(
        &exported_dense(converted.0),
        &[
            n(0., 0.),
            values[1],
            n(0., 0.),
            values[0],
            n(0., 0.),
            values[3],
            n(0., 0.),
            values[2],
            n(0., 0.),
        ],
    );
}

#[test]
fn coo_creation_export_and_duplicate_sum() {
    // Two entries at (1, 1) exercise coordinate duplicates on conversion.
    let mut rows: [Index; 4] = [1, 2, 1, 2];
    let mut cols: [Index; 4] = [1, 2, 1, 1];
    let mut values = [n(2., 1.), n(3., -1.), n(-1., 2.), n(4., 1.)];
    let mut raw = ptr::null_mut();
    unsafe {
        ok(create_coo(
            &mut raw,
            SparseIndexBase::One,
            2,
            2,
            4,
            rows.as_mut_ptr(),
            cols.as_mut_ptr(),
            values.as_mut_ptr(),
        ));
    }
    let handle = Handle(raw, PhantomData);
    let (mut base, mut nr, mut nc, mut nnz) = (SparseIndexBase::Zero, 0, 0, 0);
    let (mut ri, mut ci, mut val) = (ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
    unsafe {
        ok(export_coo(
            handle.0, &mut base, &mut nr, &mut nc, &mut nnz, &mut ri, &mut ci, &mut val,
        ));
        assert_eq!((base, nr, nc, nnz), (SparseIndexBase::One, 2, 2, 4));
        assert_eq!(std::slice::from_raw_parts(ri, 4), &rows);
        assert_eq!(std::slice::from_raw_parts(ci, 4), &cols);
        close(std::slice::from_raw_parts(val, 4), &values);
    }
    let mut converted = ptr::null_mut();
    unsafe {
        ok(mkl_sparse_convert_csr(handle.0, OP, &mut converted));
    }
    let converted = Handle(converted, PhantomData);
    close(
        &exported_dense(converted.0),
        &[values[0].add(values[2]), n(0., 0.), values[3], values[1]],
    );
}

#[test]
fn bsr_column_layout_export_and_full_update() {
    let mut offsets: [Index; 3] = [0, 1, 2];
    let mut indices: [Index; 2] = [0, 1];
    // Two 2x2 blocks, column major inside each block, one-based scalar values.
    let mut values: Vec<_> = (1..=8).map(|v| n(v as f64, (v % 3) as f64)).collect();
    // Both allocations must outlive the handle: full updates retain replacement pointers.
    let mut replacement: Vec<_> = values.iter().map(|&v| v.mul(n(2., 1.))).collect();
    let mut raw = ptr::null_mut();
    unsafe {
        ok(create_bsr(
            &mut raw,
            SparseIndexBase::Zero,
            SparseLayout::ColumnMajor,
            2,
            2,
            2,
            offsets.as_mut_ptr(),
            offsets.as_mut_ptr().add(1),
            indices.as_mut_ptr(),
            values.as_mut_ptr(),
        ));
    }
    let handle = Handle(raw, PhantomData);
    let (mut base, mut layout, mut nr, mut nc, mut bs) =
        (SparseIndexBase::One, SparseLayout::RowMajor, 0, 0, 0);
    let (mut start, mut end, mut index, mut value) = (
        ptr::null_mut(),
        ptr::null_mut(),
        ptr::null_mut(),
        ptr::null_mut(),
    );
    unsafe {
        ok(export_bsr(
            handle.0,
            &mut base,
            &mut layout,
            &mut nr,
            &mut nc,
            &mut bs,
            &mut start,
            &mut end,
            &mut index,
            &mut value,
        ));
        assert_eq!(
            (base, layout, nr, nc, bs),
            (SparseIndexBase::Zero, SparseLayout::ColumnMajor, 2, 2, 2)
        );
        assert_eq!(std::slice::from_raw_parts(start, 2), &[0, 1]);
        assert_eq!(std::slice::from_raw_parts(end, 2), &[1, 2]);
        assert_eq!(std::slice::from_raw_parts(index, 2), &indices);
        close(std::slice::from_raw_parts(value, 8), &values);
    }
    let x = [n(1., 1.), n(2., -1.), n(3., 2.), n(4., -2.)];
    let mut y = [n(0., 0.); 4];
    for update in [false, true] {
        if update {
            unsafe {
                ok(update_values(
                    handle.0,
                    8,
                    ptr::null(),
                    ptr::null(),
                    replacement.as_mut_ptr(),
                ));
            }
            unsafe {
                ok(export_bsr(
                    handle.0,
                    &mut base,
                    &mut layout,
                    &mut nr,
                    &mut nc,
                    &mut bs,
                    &mut start,
                    &mut end,
                    &mut index,
                    &mut value,
                ));
            }
            assert_eq!(
                value,
                replacement.as_mut_ptr(),
                "full BSR update retains the new values pointer"
            );
        }
        let logical_values = if update { &replacement } else { &values };
        let mut matrix = vec![n(0., 0.); 16];
        for block in 0..2 {
            for row in 0..2 {
                for col in 0..2 {
                    matrix[(block * 2 + row) * 4 + block * 2 + col] =
                        logical_values[block * 4 + col * 2 + row];
                }
            }
        }
        unsafe {
            ok(mv(
                OP,
                n(1., 0.),
                handle.0,
                GENERAL,
                x.as_ptr(),
                n(0., 0.),
                y.as_mut_ptr(),
            ));
        }
        close(&y, &product(&matrix, &x, 4, 4, 1));
    }
}

#[test]
fn matrix_multiply_all_operations_layouts_and_padding() {
    let matrix = [
        n(2., 1.),
        n(-1., 2.),
        n(3., -1.),
        n(4., 1.),
        n(0., 0.),
        n(5., -2.),
    ];
    let a = csr_matrix(&matrix, 2, 3);
    let alpha = n(2., 1.);
    let beta = n(-1., 2.);
    let sentinel = n(-999., -999.);
    for operation in [
        OP,
        SparseOperation::Transpose,
        SparseOperation::ConjugateTranspose,
    ] {
        let (rows, inner, modified) = if operation == OP {
            (2, 3, matrix.to_vec())
        } else {
            (
                3,
                2,
                transpose(
                    &matrix,
                    2,
                    3,
                    operation == SparseOperation::ConjugateTranspose,
                ),
            )
        };
        let columns = 2;
        let logical_x: Vec<_> = (0..inner * columns)
            .map(|i| n((i + 1) as f64, (i % 3) as f64))
            .collect();
        let initial = n(1., -2.);
        let expected: Vec<_> = product(&modified, &logical_x, rows, inner, columns)
            .iter()
            .map(|&v| alpha.mul(v).add(beta.mul(initial)))
            .collect();
        for layout in [SparseLayout::RowMajor, SparseLayout::ColumnMajor] {
            let ldx = if layout == SparseLayout::RowMajor {
                columns + 2
            } else {
                inner + 2
            };
            let ldy = if layout == SparseLayout::RowMajor {
                columns + 2
            } else {
                rows + 2
            };
            let mut x = vec![
                sentinel;
                if layout == SparseLayout::RowMajor {
                    inner * ldx
                } else {
                    columns * ldx
                }
            ];
            let mut y = vec![
                sentinel;
                if layout == SparseLayout::RowMajor {
                    rows * ldy
                } else {
                    columns * ldy
                }
            ];
            let at = |r: usize, c: usize, ld: usize| {
                if layout == SparseLayout::RowMajor {
                    r * ld + c
                } else {
                    c * ld + r
                }
            };
            for r in 0..inner {
                for c in 0..columns {
                    x[at(r, c, ldx)] = logical_x[r * columns + c];
                }
            }
            let mut expected_storage = y.clone();
            for r in 0..rows {
                for c in 0..columns {
                    y[at(r, c, ldy)] = initial;
                    expected_storage[at(r, c, ldy)] = expected[r * columns + c];
                }
            }
            unsafe {
                ok(mm(
                    operation,
                    alpha,
                    a.raw,
                    GENERAL,
                    layout,
                    x.as_ptr(),
                    columns as Index,
                    ldx as Index,
                    beta,
                    y.as_mut_ptr(),
                    ldy as Index,
                ));
            }
            close(&y, &expected_storage); // Padding is checked as well as numerical entries.
        }
    }
}

#[test]
fn triangular_upper_lower_unit_and_transpose_solves() {
    let solution = [n(1., 1.), n(2., -1.), n(3., 2.), n(4., -2.)];
    let alpha = n(2., 1.);
    // Divide only by alpha to form a right-hand side for a known solution.
    let (ar, ai) = alpha.parts();
    let inverse = n(ar / (ar * ar + ai * ai), -ai / (ar * ar + ai * ai));
    for mode in [SparseFillMode::Lower, SparseFillMode::Upper] {
        for diag in [SparseDiagType::NonUnit, SparseDiagType::Unit] {
            let mut stored = vec![n(0., 0.); 4];
            stored[0] = n(2., 0.);
            stored[3] = n(3., 0.);
            stored[if mode == SparseFillMode::Lower { 2 } else { 1 }] = n(1., 2.);
            let a = csr_matrix(&stored, 2, 2);
            let mut logical = stored.clone();
            if diag == SparseDiagType::Unit {
                logical[0] = n(1., 0.);
                logical[3] = n(1., 0.);
            }
            let descr = MatrixDescr {
                matrix_type: SparseMatrixType::Triangular,
                mode,
                diag,
            };
            for operation in [
                OP,
                SparseOperation::Transpose,
                SparseOperation::ConjugateTranspose,
            ] {
                let modified = if operation == OP {
                    logical.clone()
                } else {
                    transpose(
                        &logical,
                        2,
                        2,
                        operation == SparseOperation::ConjugateTranspose,
                    )
                };
                let rhs: Vec<_> = product(&modified, &solution, 2, 2, 2)
                    .iter()
                    .map(|&v| v.mul(inverse))
                    .collect();
                let mut output = vec![n(0., 0.); 4];
                unsafe {
                    ok(trsm(
                        operation,
                        alpha,
                        a.raw,
                        descr,
                        SparseLayout::RowMajor,
                        rhs.as_ptr(),
                        2,
                        2,
                        output.as_mut_ptr(),
                        2,
                    ));
                }
                close(&output, &solution);
                let rhs_vector = [rhs[0], rhs[2]];
                let mut output = [n(0., 0.); 2];
                unsafe {
                    ok(trsv(
                        operation,
                        alpha,
                        a.raw,
                        descr,
                        rhs_vector.as_ptr(),
                        output.as_mut_ptr(),
                    ));
                }
                close(&output, &[solution[0], solution[2]]);
            }
        }
    }
}

#[test]
fn symmetric_and_hermitian_descriptors() {
    for mode in [SparseFillMode::Lower, SparseFillMode::Upper] {
        let mut stored = [n(2., 0.), n(0., 0.), n(0., 0.), n(3., 0.)];
        let v = n(1., 2.);
        stored[if mode == SparseFillMode::Lower { 2 } else { 1 }] = v;
        let a = csr_matrix(&stored, 2, 2);
        let x = [n(2., 1.), n(3., -1.)];
        let kinds = if Number::COMPLEX {
            vec![SparseMatrixType::Symmetric, SparseMatrixType::Hermitian]
        } else {
            vec![SparseMatrixType::Symmetric]
        };
        for kind in kinds {
            let mut matrix = stored;
            matrix[if mode == SparseFillMode::Lower { 1 } else { 2 }] =
                if kind == SparseMatrixType::Hermitian {
                    v.conj()
                } else {
                    v
                };
            let descr = MatrixDescr {
                matrix_type: kind,
                mode,
                diag: SparseDiagType::NonUnit,
            };
            let mut y = [n(0., 0.); 2];
            unsafe {
                ok(mv(
                    OP,
                    n(1., 0.),
                    a.raw,
                    descr,
                    x.as_ptr(),
                    n(0., 0.),
                    y.as_mut_ptr(),
                ));
            }
            close(&y, &product(&matrix, &x, 2, 2, 1));
        }
    }
}

#[test]
fn addition_and_sparse_to_dense_product() {
    let matrix = [n(2., 1.), n(0., 0.), n(1., 2.), n(3., -1.)];
    let a = csr_matrix(&matrix, 2, 2);
    let alpha = n(2., 1.);
    for operation in [
        OP,
        SparseOperation::Transpose,
        SparseOperation::ConjugateTranspose,
    ] {
        let modified = if operation == OP {
            matrix.to_vec()
        } else {
            transpose(
                &matrix,
                2,
                2,
                operation == SparseOperation::ConjugateTranspose,
            )
        };
        let mut sum = ptr::null_mut();
        unsafe {
            ok(add(operation, a.raw, alpha, a.raw, &mut sum));
        }
        let sum = Handle(sum, PhantomData);
        let expected: Vec<_> = modified
            .iter()
            .zip(matrix)
            .map(|(&v, b)| alpha.mul(v).add(b))
            .collect();
        close(&exported_dense(sum.0), &expected);
        for layout in [SparseLayout::RowMajor, SparseLayout::ColumnMajor] {
            let mut result = vec![n(-999., -999.); 6];
            unsafe {
                ok(spmmd(
                    operation,
                    a.raw,
                    a.raw,
                    layout,
                    result.as_mut_ptr(),
                    3,
                ));
            }
            let expected = product(&modified, &matrix, 2, 2, 2);
            close(
                &result,
                &if layout == SparseLayout::RowMajor {
                    vec![
                        expected[0],
                        expected[1],
                        n(-999., -999.),
                        expected[2],
                        expected[3],
                        n(-999., -999.),
                    ]
                } else {
                    vec![
                        expected[0],
                        expected[2],
                        n(-999., -999.),
                        expected[1],
                        expected[3],
                        n(-999., -999.),
                    ]
                },
            );
        }
    }
}

#[test]
fn single_entry_mutation_preserves_structure() {
    let matrix = [n(2., 1.), n(0., 0.), n(1., 2.), n(3., -1.)];
    let a = csr_matrix(&matrix, 2, 2);
    let replacement = n(5., -3.);
    unsafe {
        ok(set_value(a.raw, 1, 0, replacement));
    }
    close(
        &exported_dense(a.raw),
        &[matrix[0], matrix[1], replacement, matrix[3]],
    );
}

#[test]
fn zero_nonzeros_only_scales_output() {
    let a = csr_matrix(&[n(0., 0.); 6], 2, 3);
    let x = [n(1., 2.); 3];
    let original = [n(2., -1.), n(-3., 1.)];
    let beta = n(2., 1.);
    let mut y = original;
    unsafe {
        ok(mv(
            OP,
            n(3., -1.),
            a.raw,
            GENERAL,
            x.as_ptr(),
            beta,
            y.as_mut_ptr(),
        ));
    }
    close(&y, &original.map(|v| beta.mul(v)));
    close(&exported_dense(a.raw), &[n(0., 0.); 6]);
}

#[test]
fn sparse_products_and_independent_operand_operations() {
    let matrix_a = [n(2., 1.), n(0., 0.), n(1., 2.), n(3., -1.)];
    let matrix_b = [n(1., -1.), n(4., 2.), n(0., 0.), n(2., 1.)];
    let a = csr_matrix(&matrix_a, 2, 2);
    let b = csr_matrix(&matrix_b, 2, 2);
    for op_a in [
        OP,
        SparseOperation::Transpose,
        SparseOperation::ConjugateTranspose,
    ] {
        let modified_a = if op_a == OP {
            matrix_a.to_vec()
        } else {
            transpose(&matrix_a, 2, 2, op_a == SparseOperation::ConjugateTranspose)
        };
        let mut raw = ptr::null_mut();
        unsafe {
            ok(mkl_sparse_spmm(op_a, a.raw, b.raw, &mut raw));
        }
        let result = Handle(raw, PhantomData);
        close(
            &exported_dense(result.0),
            &product(&modified_a, &matrix_b, 2, 2, 2),
        );
        for op_b in [
            OP,
            SparseOperation::Transpose,
            SparseOperation::ConjugateTranspose,
        ] {
            let modified_b = if op_b == OP {
                matrix_b.to_vec()
            } else {
                transpose(&matrix_b, 2, 2, op_b == SparseOperation::ConjugateTranspose)
            };
            let mut raw = ptr::null_mut();
            unsafe {
                ok(mkl_sparse_sp2m(
                    op_a,
                    GENERAL,
                    a.raw,
                    op_b,
                    GENERAL,
                    b.raw,
                    SparseRequest::FullMult,
                    &mut raw,
                ));
            }
            let result = Handle(raw, PhantomData);
            close(
                &exported_dense(result.0),
                &product(&modified_a, &modified_b, 2, 2, 2),
            );
        }
    }
}
