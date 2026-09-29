use std::{
    marker::PhantomData,
    mem::{align_of, offset_of, size_of},
    ptr,
};

const OP: SparseOperation = SparseOperation::NonTranspose;
const GENERAL: MatrixDescr = MatrixDescr {
    matrix_type: SparseMatrixType::General,
    mode: SparseFillMode::Lower,
    diag: SparseDiagType::NonUnit,
};
const TRIANGULAR: MatrixDescr = MatrixDescr {
    matrix_type: SparseMatrixType::Triangular,
    ..GENERAL
};

// Test-only owner. The borrow marker keeps creation arrays alive until destruction.
struct Handle<'a>(SparseMatrixHandle, PhantomData<&'a mut ()>);
impl Drop for Handle<'_> {
    fn drop(&mut self) {
        let status = unsafe { mkl_sparse_destroy(self.0) };
        if !std::thread::panicking() {
            assert_eq!(status, SparseStatus::SUCCESS);
        }
    }
}

fn csr<'a>(
    offsets: &'a mut [Index; 3],
    indices: &'a mut [Index; 3],
    values: &'a mut [f64; 3],
) -> Handle<'a> {
    let mut a = ptr::null_mut();
    unsafe {
        assert_eq!(
            mkl_sparse_d_create_csr(
                &mut a,
                SparseIndexBase::Zero,
                2,
                2,
                offsets.as_mut_ptr(),
                offsets.as_mut_ptr().add(1),
                indices.as_mut_ptr(),
                values.as_mut_ptr()
            ),
            SparseStatus::SUCCESS
        );
    }
    Handle(a, PhantomData)
}

fn dense(a: SparseMatrixHandle) -> [f64; 4] {
    let identity = [1., 0., 0., 1.];
    let mut result = [0.; 4];
    unsafe {
        assert_eq!(
            mkl_sparse_d_mm(
                OP,
                1.,
                a,
                GENERAL,
                SparseLayout::RowMajor,
                identity.as_ptr(),
                2,
                2,
                0.,
                result.as_mut_ptr(),
                2
            ),
            SparseStatus::SUCCESS
        );
    }
    result
}

#[test]
fn c_layout_and_integer_widths() {
    assert_eq!(size_of::<SparseMatrixHandle>(), size_of::<usize>());
    assert_eq!(size_of::<SparseStatus>(), 4);
    assert_eq!(size_of::<SparseOperation>(), 4);
    assert_eq!(size_of::<MatrixDescr>(), 12);
    assert_eq!(offset_of!(MatrixDescr, mode), 4);
    assert_eq!(offset_of!(MatrixDescr, diag), 8);
    assert_eq!(
        (size_of::<MklComplex8>(), align_of::<MklComplex8>()),
        (8, 4)
    );
    assert_eq!(
        (size_of::<MklComplex16>(), align_of::<MklComplex16>()),
        (16, 8)
    );
    assert_eq!(offset_of!(MklComplex16, imag), 8);
    assert_eq!(size_of::<MklInt64>(), 8);
}

#[test]
fn csr_vector_transpose_and_beta() {
    let mut offsets = [0, 1, 3];
    let mut indices = [0, 0, 1];
    let mut values = [2., 1., 3.];
    let a = csr(&mut offsets, &mut indices, &mut values);
    let x = [1., 2.];
    let mut y = [5., 7.];
    unsafe {
        assert_eq!(
            mkl_sparse_set_mv_hint(a.0, OP, GENERAL, 2),
            SparseStatus::SUCCESS
        );
        assert_eq!(
            mkl_sparse_set_memory_hint(a.0, SparseMemoryUsage::None),
            SparseStatus::SUCCESS
        );
        assert_eq!(mkl_sparse_optimize(a.0), SparseStatus::SUCCESS);
        assert_eq!(
            mkl_sparse_d_mv(OP, 2., a.0, GENERAL, x.as_ptr(), 3., y.as_mut_ptr()),
            SparseStatus::SUCCESS
        );
        assert_eq!(y, [19., 35.]);
        assert_eq!(
            mkl_sparse_d_mv(
                SparseOperation::Transpose,
                1.,
                a.0,
                GENERAL,
                x.as_ptr(),
                0.,
                y.as_mut_ptr()
            ),
            SparseStatus::SUCCESS
        );
    }
    assert_eq!(y, [4., 6.]);
}

#[test]
fn single_precision_csr() {
    let mut offsets: [Index; 3] = [0, 1, 2];
    let mut indices: [Index; 2] = [0, 1];
    let mut values = [2_f32, 3.];
    let mut raw = ptr::null_mut();
    let x = [4_f32, 5.];
    let mut y = [0_f32; 2];
    unsafe {
        assert_eq!(
            mkl_sparse_s_create_csr(
                &mut raw,
                SparseIndexBase::Zero,
                2,
                2,
                offsets.as_mut_ptr(),
                offsets.as_mut_ptr().add(1),
                indices.as_mut_ptr(),
                values.as_mut_ptr()
            ),
            SparseStatus::SUCCESS
        );
    }
    let a = Handle(raw, PhantomData);
    unsafe {
        assert_eq!(
            mkl_sparse_s_mv(OP, 1., a.0, GENERAL, x.as_ptr(), 0., y.as_mut_ptr()),
            SparseStatus::SUCCESS
        );
    }
    assert_eq!(y, [8., 15.]);
}

macro_rules! complex_case {
    ($name:ident, $scalar:ident, $create:ident, $mv:ident) => {
        #[test]
        fn $name() {
            let mut offsets: [Index; 3] = [0, 1, 2];
            let mut indices: [Index; 2] = [0, 1];
            let mut values = [
                $scalar { real: 2., imag: 1. },
                $scalar {
                    real: 3.,
                    imag: -1.,
                },
            ];
            let x = [
                $scalar { real: 1., imag: 1. },
                $scalar { real: 2., imag: 0. },
            ];
            let mut y = [$scalar { real: 1., imag: 0. }; 2];
            let mut raw = ptr::null_mut();
            unsafe {
                assert_eq!(
                    $create(
                        &mut raw,
                        SparseIndexBase::Zero,
                        2,
                        2,
                        offsets.as_mut_ptr(),
                        offsets.as_mut_ptr().add(1),
                        indices.as_mut_ptr(),
                        values.as_mut_ptr()
                    ),
                    SparseStatus::SUCCESS
                );
            }
            let a = Handle(raw, PhantomData);
            unsafe {
                assert_eq!(
                    $mv(
                        OP,
                        $scalar { real: 0., imag: 1. },
                        a.0,
                        GENERAL,
                        x.as_ptr(),
                        $scalar { real: 2., imag: 0. },
                        y.as_mut_ptr()
                    ),
                    SparseStatus::SUCCESS
                );
            }
            assert_eq!(
                y,
                [
                    $scalar {
                        real: -1.,
                        imag: 1.
                    },
                    $scalar { real: 4., imag: 6. }
                ]
            );
            unsafe {
                assert_eq!(
                    $mv(
                        SparseOperation::ConjugateTranspose,
                        $scalar { real: 1., imag: 0. },
                        a.0,
                        GENERAL,
                        x.as_ptr(),
                        $scalar { real: 0., imag: 0. },
                        y.as_mut_ptr()
                    ),
                    SparseStatus::SUCCESS
                );
            }
            assert_eq!(
                y,
                [
                    $scalar { real: 3., imag: 1. },
                    $scalar { real: 6., imag: 2. }
                ]
            );
        }
    };
}
complex_case!(
    complex_single_by_value,
    MklComplex8,
    mkl_sparse_c_create_csr,
    mkl_sparse_c_mv
);
complex_case!(
    complex_double_by_value,
    MklComplex16,
    mkl_sparse_z_create_csr,
    mkl_sparse_z_mv
);

#[test]
fn dense_matrix_layouts_and_triangular_solves() {
    let mut offsets = [0, 1, 3];
    let mut indices = [0, 0, 1];
    let mut values = [2., 1., 3.];
    let a = csr(&mut offsets, &mut indices, &mut values);
    unsafe {
        assert_eq!(
            mkl_sparse_set_mm_hint(a.0, OP, GENERAL, SparseLayout::RowMajor, 2, 3),
            SparseStatus::SUCCESS
        );
        assert_eq!(
            mkl_sparse_set_sv_hint(a.0, OP, TRIANGULAR, 2),
            SparseStatus::SUCCESS
        );
        assert_eq!(
            mkl_sparse_set_sm_hint(a.0, OP, TRIANGULAR, SparseLayout::RowMajor, 2, 2),
            SparseStatus::SUCCESS
        );
        assert_eq!(mkl_sparse_optimize(a.0), SparseStatus::SUCCESS);
    }
    for (layout, x, expected) in [
        (SparseLayout::RowMajor, [1., 2., 3., 4.], [4., 6., 12., 16.]),
        (
            SparseLayout::ColumnMajor,
            [1., 3., 2., 4.],
            [4., 12., 6., 16.],
        ),
    ] {
        let mut y = [1.; 4];
        unsafe {
            assert_eq!(
                mkl_sparse_d_mm(
                    OP,
                    1.,
                    a.0,
                    GENERAL,
                    layout,
                    x.as_ptr(),
                    2,
                    2,
                    2.,
                    y.as_mut_ptr(),
                    2
                ),
                SparseStatus::SUCCESS
            );
        }
        assert_eq!(y, expected);
        let rhs = y.map(|v| v - 2.);
        unsafe {
            assert_eq!(
                mkl_sparse_d_trsm(
                    OP,
                    1.,
                    a.0,
                    TRIANGULAR,
                    layout,
                    rhs.as_ptr(),
                    2,
                    2,
                    y.as_mut_ptr(),
                    2
                ),
                SparseStatus::SUCCESS
            );
        }
        assert_eq!(y, x);
    }
    let rhs = [2., 7.];
    let mut y = [0.; 2];
    unsafe {
        assert_eq!(
            mkl_sparse_d_trsv(OP, 1., a.0, TRIANGULAR, rhs.as_ptr(), y.as_mut_ptr()),
            SparseStatus::SUCCESS
        );
    }
    assert_eq!(y, [1., 2.]);
}

#[test]
fn one_based_csc_and_coo_conversion() {
    let mut offsets: [Index; 3] = [1, 3, 4];
    let mut indices: [Index; 3] = [1, 2, 2];
    let mut values = [2., 1., 3.];
    let mut raw = ptr::null_mut();
    unsafe {
        assert_eq!(
            mkl_sparse_d_create_csc(
                &mut raw,
                SparseIndexBase::One,
                2,
                2,
                offsets.as_mut_ptr(),
                offsets.as_mut_ptr().add(1),
                indices.as_mut_ptr(),
                values.as_mut_ptr()
            ),
            SparseStatus::SUCCESS
        );
    }
    let a = Handle(raw, PhantomData);
    let mut converted = ptr::null_mut();
    unsafe {
        assert_eq!(
            mkl_sparse_convert_csr(a.0, OP, &mut converted),
            SparseStatus::SUCCESS
        );
    }
    let converted = Handle(converted, PhantomData);
    assert_eq!(dense(converted.0), [2., 0., 1., 3.]);

    let mut row: [Index; 3] = [0, 1, 1];
    let mut col: [Index; 3] = [0, 0, 1];
    let mut val = [2., 1., 3.];
    let mut coo = ptr::null_mut();
    unsafe {
        assert_eq!(
            mkl_sparse_d_create_coo(
                &mut coo,
                SparseIndexBase::Zero,
                2,
                2,
                3,
                row.as_mut_ptr(),
                col.as_mut_ptr(),
                val.as_mut_ptr()
            ),
            SparseStatus::SUCCESS
        );
    }
    let coo = Handle(coo, PhantomData);
    let mut converted = ptr::null_mut();
    unsafe {
        assert_eq!(
            mkl_sparse_convert_csr(coo.0, OP, &mut converted),
            SparseStatus::SUCCESS
        );
    }
    let converted = Handle(converted, PhantomData);
    assert_eq!(dense(converted.0), [2., 0., 1., 3.]);
}

#[test]
fn bsr_block_dimensions_and_export() {
    let mut offsets: [Index; 2] = [0, 1];
    let mut indices: [Index; 1] = [0];
    let mut values = [2., 0., 1., 3.];
    let mut raw = ptr::null_mut();
    unsafe {
        assert_eq!(
            mkl_sparse_d_create_bsr(
                &mut raw,
                SparseIndexBase::Zero,
                SparseLayout::RowMajor,
                1,
                1,
                2,
                offsets.as_mut_ptr(),
                offsets.as_mut_ptr().add(1),
                indices.as_mut_ptr(),
                values.as_mut_ptr()
            ),
            SparseStatus::SUCCESS
        );
    }
    let a = Handle(raw, PhantomData);
    let mut y = [0.; 2];
    let x = [1., 2.];
    unsafe {
        assert_eq!(
            mkl_sparse_d_mv(OP, 1., a.0, GENERAL, x.as_ptr(), 0., y.as_mut_ptr()),
            SparseStatus::SUCCESS
        );
    }
    assert_eq!(y, [2., 7.]);
    let (mut base, mut layout) = (SparseIndexBase::One, SparseLayout::ColumnMajor);
    let (mut rows, mut cols, mut block_size) = (0, 0, 0);
    let (mut start, mut end, mut index, mut value) = (
        ptr::null_mut(),
        ptr::null_mut(),
        ptr::null_mut(),
        ptr::null_mut(),
    );
    unsafe {
        assert_eq!(
            mkl_sparse_d_export_bsr(
                a.0,
                &mut base,
                &mut layout,
                &mut rows,
                &mut cols,
                &mut block_size,
                &mut start,
                &mut end,
                &mut index,
                &mut value
            ),
            SparseStatus::SUCCESS
        );
        assert_eq!(
            (base, layout, rows, cols, block_size),
            (SparseIndexBase::Zero, SparseLayout::RowMajor, 1, 1, 2)
        );
        assert_eq!((*start, *end, *index), (0, 1, 0));
        assert_eq!(std::slice::from_raw_parts(value, 4), &[2., 0., 1., 3.]);
    }
}

#[test]
fn ordering_export_and_value_updates() {
    let mut offsets = [0, 1, 3];
    let mut indices = [0, 1, 0];
    let mut values = [2., 3., 1.];
    let a = csr(&mut offsets, &mut indices, &mut values);
    unsafe {
        assert_eq!(mkl_sparse_order(a.0), SparseStatus::SUCCESS);
        let (mut base, mut rows, mut cols) = (SparseIndexBase::One, 0, 0);
        let (mut start, mut end, mut index, mut value) = (
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
        );
        assert_eq!(
            mkl_sparse_d_export_csr(
                a.0, &mut base, &mut rows, &mut cols, &mut start, &mut end, &mut index, &mut value
            ),
            SparseStatus::SUCCESS
        );
        assert_eq!((base, rows, cols), (SparseIndexBase::Zero, 2, 2));
        assert_eq!(std::slice::from_raw_parts(start, 2), &[0, 1]);
        assert_eq!(std::slice::from_raw_parts(end, 2), &[1, 3]);
        assert_eq!(std::slice::from_raw_parts(index, 3), &[0, 0, 1]);
        assert_eq!(std::slice::from_raw_parts(value, 3), &[2., 1., 3.]);
        assert_eq!(mkl_sparse_d_set_value(a.0, 1, 0, 4.), SparseStatus::SUCCESS);
        let row: [Index; 1] = [0];
        let col: [Index; 1] = [0];
        let mut val = [5.];
        assert_eq!(
            mkl_sparse_d_update_values(a.0, 1, row.as_ptr(), col.as_ptr(), val.as_mut_ptr()),
            SparseStatus::NOT_SUPPORTED
        );
        assert_eq!(mkl_sparse_d_set_value(a.0, 0, 0, 5.), SparseStatus::SUCCESS);
    }
    assert_eq!(dense(a.0), [5., 0., 4., 3.]);
    drop(a);
    // MKL mutates caller-owned storage; destruction must not free these arrays.
    assert_eq!(indices, [0, 0, 1]);
    assert_eq!(values, [5., 4., 3.]);
}

#[test]
fn copy_add_and_sparse_products() {
    let mut offsets = [0, 1, 3];
    let mut indices = [0, 0, 1];
    let mut values = [2., 1., 3.];
    let a = csr(&mut offsets, &mut indices, &mut values);
    let mut raw = ptr::null_mut();
    unsafe {
        assert_eq!(
            mkl_sparse_copy(a.0, GENERAL, &mut raw),
            SparseStatus::SUCCESS
        );
    }
    let copied = Handle(raw, PhantomData);
    assert_eq!(dense(copied.0), [2., 0., 1., 3.]);
    let mut raw = ptr::null_mut();
    unsafe {
        assert_eq!(
            mkl_sparse_d_add(OP, a.0, 2., copied.0, &mut raw),
            SparseStatus::SUCCESS
        );
    }
    let sum = Handle(raw, PhantomData);
    assert_eq!(dense(sum.0), [6., 0., 3., 9.]);
    let mut raw = ptr::null_mut();
    unsafe {
        assert_eq!(
            mkl_sparse_spmm(OP, a.0, copied.0, &mut raw),
            SparseStatus::SUCCESS
        );
    }
    let product = Handle(raw, PhantomData);
    assert_eq!(dense(product.0), [4., 0., 5., 9.]);
    let mut result = [0.; 4];
    unsafe {
        assert_eq!(
            mkl_sparse_d_spmmd(
                OP,
                a.0,
                copied.0,
                SparseLayout::RowMajor,
                result.as_mut_ptr(),
                2
            ),
            SparseStatus::SUCCESS
        );
    }
    assert_eq!(result, [4., 0., 5., 9.]);
}

#[test]
fn staged_product_and_recompute_values() {
    let mut offsets = [0, 1, 3];
    let mut indices = [0, 0, 1];
    let mut values = [2., 1., 3.];
    let a = csr(&mut offsets, &mut indices, &mut values);
    let mut raw = ptr::null_mut();
    unsafe {
        assert_eq!(
            mkl_sparse_sp2m(
                OP,
                GENERAL,
                a.0,
                OP,
                GENERAL,
                a.0,
                SparseRequest::NnzCount,
                &mut raw
            ),
            SparseStatus::SUCCESS
        );
    }
    let mut product = Handle(raw, PhantomData);
    unsafe {
        let (mut base, mut rows, mut cols) = (SparseIndexBase::Zero, 0, 0);
        let (mut start, mut end, mut index, mut value) = (
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
        );
        assert_eq!(
            mkl_sparse_d_export_csr(
                product.0, &mut base, &mut rows, &mut cols, &mut start, &mut end, &mut index,
                &mut value
            ),
            SparseStatus::SUCCESS
        );
        assert_eq!((rows, cols), (2, 2));
        assert_eq!(*end.add(1) - *start, 3);
        // Indices and values are not read before finalization.
        assert_eq!(
            mkl_sparse_sp2m(
                OP,
                GENERAL,
                a.0,
                OP,
                GENERAL,
                a.0,
                SparseRequest::FinalizeMult,
                &mut product.0
            ),
            SparseStatus::SUCCESS
        );
    }
    assert_eq!(dense(product.0), [4., 0., 5., 9.]);
    unsafe {
        assert_eq!(mkl_sparse_d_set_value(a.0, 0, 0, 4.), SparseStatus::SUCCESS);
        assert_eq!(
            mkl_sparse_sp2m(
                OP,
                GENERAL,
                a.0,
                OP,
                GENERAL,
                a.0,
                SparseRequest::FinalizeMult,
                &mut product.0
            ),
            SparseStatus::SUCCESS
        );
    }
    assert_eq!(dense(product.0), [16., 0., 7., 9.]);
}

#[test]
fn empty_handle_returns_status() {
    let x = [1.];
    let mut y = [0.];
    let status = unsafe {
        mkl_sparse_d_mv(
            OP,
            1.,
            ptr::null_mut(),
            GENERAL,
            x.as_ptr(),
            0.,
            y.as_mut_ptr(),
        )
    };
    assert_eq!(status, SparseStatus::NOT_INITIALIZED);
}

#[test]
fn bsr_full_value_update() {
    let mut offsets: [Index; 2] = [0, 1];
    let mut indices: [Index; 1] = [0];
    let mut values = [2., 0., 1., 3.];
    let mut raw = ptr::null_mut();
    unsafe {
        assert_eq!(
            mkl_sparse_d_create_bsr(
                &mut raw,
                SparseIndexBase::Zero,
                SparseLayout::RowMajor,
                1,
                1,
                2,
                offsets.as_mut_ptr(),
                offsets.as_mut_ptr().add(1),
                indices.as_mut_ptr(),
                values.as_mut_ptr()
            ),
            SparseStatus::SUCCESS
        );
    }
    let a = Handle(raw, PhantomData);
    let mut replacement = [4., 0., 2., 6.];
    let x = [1., 2.];
    let mut y = [0.; 2];
    unsafe {
        assert_eq!(
            mkl_sparse_d_update_values(a.0, 0, ptr::null(), ptr::null(), replacement.as_mut_ptr()),
            SparseStatus::SUCCESS
        );
        assert_eq!(
            mkl_sparse_d_mv(OP, 1., a.0, GENERAL, x.as_ptr(), 0., y.as_mut_ptr()),
            SparseStatus::SUCCESS
        );
    }
    assert_eq!(y, [4., 14.]);
}

#[test]
fn rectangular_transpose_and_padded_dense_storage() {
    let mut offsets: [Index; 3] = [0, 2, 3];
    let mut indices: [Index; 3] = [0, 2, 1];
    let mut values = [2., 1., 3.];
    let mut raw = ptr::null_mut();
    unsafe {
        assert_eq!(
            mkl_sparse_d_create_csr(
                &mut raw,
                SparseIndexBase::Zero,
                2,
                3,
                offsets.as_mut_ptr(),
                offsets.as_mut_ptr().add(1),
                indices.as_mut_ptr(),
                values.as_mut_ptr()
            ),
            SparseStatus::SUCCESS
        );
    }
    let a = Handle(raw, PhantomData);
    let x = [4., 5.];
    let mut y = [0.; 3];
    unsafe {
        assert_eq!(
            mkl_sparse_d_mv(
                SparseOperation::Transpose,
                1.,
                a.0,
                GENERAL,
                x.as_ptr(),
                0.,
                y.as_mut_ptr()
            ),
            SparseStatus::SUCCESS
        );
    }
    assert_eq!(y, [8., 15., 4.]);
    // Three logical input rows, two columns per row, with a padded stride of 3.
    let x = [1., 2., -999., 3., 4., -999., 5., 6., -999.];
    let mut y = [-999.; 6];
    unsafe {
        assert_eq!(
            mkl_sparse_d_mm(
                OP,
                1.,
                a.0,
                GENERAL,
                SparseLayout::RowMajor,
                x.as_ptr(),
                2,
                3,
                0.,
                y.as_mut_ptr(),
                3
            ),
            SparseStatus::SUCCESS
        );
    }
    assert_eq!(y, [7., 10., -999., 9., 12., -999.]);
}

#[test]
fn format_conversion_round_trips_and_transpose() {
    let mut offsets = [0, 1, 3];
    let mut indices = [0, 0, 1];
    let mut values = [2., 1., 3.];
    let a = csr(&mut offsets, &mut indices, &mut values);
    for operation in [OP, SparseOperation::Transpose] {
        for format in [0, 1, 2] {
            let mut raw = ptr::null_mut();
            unsafe {
                let status = match format {
                    0 => mkl_sparse_convert_csc(a.0, operation, &mut raw),
                    1 => mkl_sparse_convert_coo(a.0, operation, &mut raw),
                    _ => mkl_sparse_convert_bsr(
                        a.0,
                        2,
                        SparseLayout::ColumnMajor,
                        operation,
                        &mut raw,
                    ),
                };
                assert_eq!(status, SparseStatus::SUCCESS);
            }
            let converted = Handle(raw, PhantomData);
            let mut raw = ptr::null_mut();
            unsafe {
                assert_eq!(
                    mkl_sparse_convert_csr(converted.0, OP, &mut raw),
                    SparseStatus::SUCCESS
                );
            }
            let round_trip = Handle(raw, PhantomData);
            assert_eq!(
                dense(round_trip.0),
                if operation == OP {
                    [2., 0., 1., 3.]
                } else {
                    [2., 1., 0., 3.]
                }
            );
        }
    }
}

#[test]
fn copied_matrix_outlives_source_and_ignores_descriptor() {
    let copied = {
        let mut offsets = [0, 1, 3];
        let mut indices = [0, 0, 1];
        let mut values = [2., 1., 3.];
        let a = csr(&mut offsets, &mut indices, &mut values);
        let mut raw = ptr::null_mut();
        unsafe {
            // Current MKL deep-copies the complete matrix and ignores descr.
            assert_eq!(
                mkl_sparse_copy(
                    a.0,
                    MatrixDescr {
                        matrix_type: SparseMatrixType::Diagonal,
                        ..GENERAL
                    },
                    &mut raw
                ),
                SparseStatus::SUCCESS
            );
            assert_eq!(
                mkl_sparse_d_set_value(a.0, 1, 0, 99.),
                SparseStatus::SUCCESS
            );
        }
        Handle(raw, PhantomData)
        // a is destroyed, then its arrays leave scope; copied owns independent storage.
    };
    assert_eq!(dense(copied.0), [2., 0., 1., 3.]);
}

#[test]
fn aggressive_inspection_repeated_execution_and_cleanup() {
    for _ in 0..64 {
        let mut offsets = [0, 1, 3];
        let mut indices = [0, 0, 1];
        let mut values = [2., 1., 3.];
        let a = csr(&mut offsets, &mut indices, &mut values);
        unsafe {
            assert_eq!(
                mkl_sparse_set_memory_hint(a.0, SparseMemoryUsage::Aggressive),
                SparseStatus::SUCCESS
            );
            assert_eq!(
                mkl_sparse_set_mv_hint(a.0, OP, GENERAL, 32),
                SparseStatus::SUCCESS
            );
            assert_eq!(mkl_sparse_optimize(a.0), SparseStatus::SUCCESS);
        }
        let x = [1., 2.];
        let mut y = [0.; 2];
        for _ in 0..32 {
            unsafe {
                assert_eq!(
                    mkl_sparse_d_mv(OP, 1., a.0, GENERAL, x.as_ptr(), 0., y.as_mut_ptr()),
                    SparseStatus::SUCCESS
                );
            }
            assert_eq!(y, [2., 7.]);
        }
    }
}

#[test]
fn full_bsr_update_on_owned_copy_keeps_caller_storage_owned_by_caller() {
    let mut offsets: [Index; 2] = [0, 1];
    let mut indices: [Index; 1] = [0];
    let mut values = [2., 0., 1., 3.];
    let mut replacement = vec![4., 0., 2., 6.];
    let mut raw = ptr::null_mut();
    unsafe {
        assert_eq!(
            mkl_sparse_d_create_bsr(
                &mut raw,
                SparseIndexBase::Zero,
                SparseLayout::RowMajor,
                1,
                1,
                2,
                offsets.as_mut_ptr(),
                offsets.as_mut_ptr().add(1),
                indices.as_mut_ptr(),
                values.as_mut_ptr()
            ),
            SparseStatus::SUCCESS
        );
    }
    let source = Handle(raw, PhantomData);
    let mut raw = ptr::null_mut();
    unsafe {
        assert_eq!(
            mkl_sparse_copy(source.0, GENERAL, &mut raw),
            SparseStatus::SUCCESS
        );
    }
    let copied = Handle(raw, PhantomData);
    drop(source);
    let x = [1., 2.];
    let mut y = [0.; 2];
    unsafe {
        assert_eq!(
            mkl_sparse_d_update_values(
                copied.0,
                0,
                ptr::null(),
                ptr::null(),
                replacement.as_mut_ptr()
            ),
            SparseStatus::SUCCESS
        );
        assert_eq!(
            mkl_sparse_d_mv(OP, 1., copied.0, GENERAL, x.as_ptr(), 0., y.as_mut_ptr()),
            SparseStatus::SUCCESS
        );
    }
    assert_eq!(y, [4., 14.]);
    drop(copied);
    // Destruction must not release caller-owned replacement storage.
    assert_eq!(replacement, [4., 0., 2., 6.]);
    replacement[0] = 9.;
    assert_eq!(replacement[0], 9.);
}
