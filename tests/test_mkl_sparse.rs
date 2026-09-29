#![cfg(rivet_blas_mkl)]

mod reference {
    include!("support/mkl_sparse_reference.rs");
}

mod ordinary {
    use rivet_blas_sys::mkl::sparse::*;
    use rivet_blas_sys::mkl::types::*;
    type Index = MklInt;
    include!("support/mkl_sparse_cases.rs");
    mod precision_s {
        use super::*;
        type Number = f32;
        use rivet_blas_sys::mkl::sparse::{
            mkl_sparse_s_add as add, mkl_sparse_s_create_bsr as create_bsr,
            mkl_sparse_s_create_coo as create_coo, mkl_sparse_s_create_csc as create_csc,
            mkl_sparse_s_create_csr as create_csr, mkl_sparse_s_export_bsr as export_bsr,
            mkl_sparse_s_export_coo as export_coo, mkl_sparse_s_export_csc as export_csc,
            mkl_sparse_s_export_csr as export_csr, mkl_sparse_s_mm as mm, mkl_sparse_s_mv as mv,
            mkl_sparse_s_set_value as set_value, mkl_sparse_s_spmmd as spmmd,
            mkl_sparse_s_trsm as trsm, mkl_sparse_s_trsv as trsv,
            mkl_sparse_s_update_values as update_values,
        };
        include!("support/mkl_sparse_precision_cases.rs");
    }

    mod precision_d {
        use super::*;
        type Number = f64;
        use rivet_blas_sys::mkl::sparse::{
            mkl_sparse_d_add as add, mkl_sparse_d_create_bsr as create_bsr,
            mkl_sparse_d_create_coo as create_coo, mkl_sparse_d_create_csc as create_csc,
            mkl_sparse_d_create_csr as create_csr, mkl_sparse_d_export_bsr as export_bsr,
            mkl_sparse_d_export_coo as export_coo, mkl_sparse_d_export_csc as export_csc,
            mkl_sparse_d_export_csr as export_csr, mkl_sparse_d_mm as mm, mkl_sparse_d_mv as mv,
            mkl_sparse_d_set_value as set_value, mkl_sparse_d_spmmd as spmmd,
            mkl_sparse_d_trsm as trsm, mkl_sparse_d_trsv as trsv,
            mkl_sparse_d_update_values as update_values,
        };
        include!("support/mkl_sparse_precision_cases.rs");
    }

    mod precision_c {
        use super::*;
        type Number = MklComplex8;
        use rivet_blas_sys::mkl::sparse::{
            mkl_sparse_c_add as add, mkl_sparse_c_create_bsr as create_bsr,
            mkl_sparse_c_create_coo as create_coo, mkl_sparse_c_create_csc as create_csc,
            mkl_sparse_c_create_csr as create_csr, mkl_sparse_c_export_bsr as export_bsr,
            mkl_sparse_c_export_coo as export_coo, mkl_sparse_c_export_csc as export_csc,
            mkl_sparse_c_export_csr as export_csr, mkl_sparse_c_mm as mm, mkl_sparse_c_mv as mv,
            mkl_sparse_c_set_value as set_value, mkl_sparse_c_spmmd as spmmd,
            mkl_sparse_c_trsm as trsm, mkl_sparse_c_trsv as trsv,
            mkl_sparse_c_update_values as update_values,
        };
        include!("support/mkl_sparse_precision_cases.rs");
    }

    mod precision_z {
        use super::*;
        type Number = MklComplex16;
        use rivet_blas_sys::mkl::sparse::{
            mkl_sparse_z_add as add, mkl_sparse_z_create_bsr as create_bsr,
            mkl_sparse_z_create_coo as create_coo, mkl_sparse_z_create_csc as create_csc,
            mkl_sparse_z_create_csr as create_csr, mkl_sparse_z_export_bsr as export_bsr,
            mkl_sparse_z_export_coo as export_coo, mkl_sparse_z_export_csc as export_csc,
            mkl_sparse_z_export_csr as export_csr, mkl_sparse_z_mm as mm, mkl_sparse_z_mv as mv,
            mkl_sparse_z_set_value as set_value, mkl_sparse_z_spmmd as spmmd,
            mkl_sparse_z_trsm as trsm, mkl_sparse_z_trsv as trsv,
            mkl_sparse_z_update_values as update_values,
        };
        include!("support/mkl_sparse_precision_cases.rs");
    }
}

mod explicit_64 {
    use rivet_blas_sys::mkl::types::*;
    type Index = MklInt64;
    // Use the same behavioral tests with the actual suffixed symbols.
    use rivet_blas_sys::mkl::sparse_64::{
        mkl_sparse_c_create_csr_64 as mkl_sparse_c_create_csr,
        mkl_sparse_c_mv_64 as mkl_sparse_c_mv, mkl_sparse_convert_bsr_64 as mkl_sparse_convert_bsr,
        mkl_sparse_convert_coo_64 as mkl_sparse_convert_coo,
        mkl_sparse_convert_csc_64 as mkl_sparse_convert_csc,
        mkl_sparse_convert_csr_64 as mkl_sparse_convert_csr, mkl_sparse_copy_64 as mkl_sparse_copy,
        mkl_sparse_d_add_64 as mkl_sparse_d_add,
        mkl_sparse_d_create_bsr_64 as mkl_sparse_d_create_bsr,
        mkl_sparse_d_create_coo_64 as mkl_sparse_d_create_coo,
        mkl_sparse_d_create_csc_64 as mkl_sparse_d_create_csc,
        mkl_sparse_d_create_csr_64 as mkl_sparse_d_create_csr,
        mkl_sparse_d_export_bsr_64 as mkl_sparse_d_export_bsr,
        mkl_sparse_d_export_csr_64 as mkl_sparse_d_export_csr,
        mkl_sparse_d_mm_64 as mkl_sparse_d_mm, mkl_sparse_d_mv_64 as mkl_sparse_d_mv,
        mkl_sparse_d_set_value_64 as mkl_sparse_d_set_value,
        mkl_sparse_d_spmmd_64 as mkl_sparse_d_spmmd, mkl_sparse_d_trsm_64 as mkl_sparse_d_trsm,
        mkl_sparse_d_trsv_64 as mkl_sparse_d_trsv,
        mkl_sparse_d_update_values_64 as mkl_sparse_d_update_values,
        mkl_sparse_destroy_64 as mkl_sparse_destroy, mkl_sparse_optimize_64 as mkl_sparse_optimize,
        mkl_sparse_order_64 as mkl_sparse_order,
        mkl_sparse_s_create_csr_64 as mkl_sparse_s_create_csr,
        mkl_sparse_s_mv_64 as mkl_sparse_s_mv,
        mkl_sparse_set_memory_hint_64 as mkl_sparse_set_memory_hint,
        mkl_sparse_set_mm_hint_64 as mkl_sparse_set_mm_hint,
        mkl_sparse_set_mv_hint_64 as mkl_sparse_set_mv_hint,
        mkl_sparse_set_sm_hint_64 as mkl_sparse_set_sm_hint,
        mkl_sparse_set_sv_hint_64 as mkl_sparse_set_sv_hint, mkl_sparse_sp2m_64 as mkl_sparse_sp2m,
        mkl_sparse_spmm_64 as mkl_sparse_spmm,
        mkl_sparse_z_create_csr_64 as mkl_sparse_z_create_csr,
        mkl_sparse_z_mv_64 as mkl_sparse_z_mv,
    };

    #[test]
    fn dimensions_and_indices_exceed_lp64_range() {
        // COO creation/export needs one value, rather than a huge dense allocation.
        let cols: MklInt64 = i32::MAX as MklInt64 + 17;
        let mut row = [0_i64];
        let mut col = [cols - 1];
        let mut values = [7.0_f64];
        let mut raw = std::ptr::null_mut();
        unsafe {
            assert_eq!(
                mkl_sparse_d_create_coo(
                    &mut raw,
                    SparseIndexBase::Zero,
                    1,
                    cols,
                    1,
                    row.as_mut_ptr(),
                    col.as_mut_ptr(),
                    values.as_mut_ptr()
                ),
                SparseStatus::SUCCESS
            );
        }
        let handle = Handle(raw, std::marker::PhantomData);
        let mut base = SparseIndexBase::One;
        let (mut rows, mut columns, mut nnz) = (0_i64, 0_i64, 0_i64);
        let (mut row_ptr, mut col_ptr, mut value_ptr) = (
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        unsafe {
            assert_eq!(
                rivet_blas_sys::mkl::sparse_64::mkl_sparse_d_export_coo_64(
                    handle.0,
                    &mut base,
                    &mut rows,
                    &mut columns,
                    &mut nnz,
                    &mut row_ptr,
                    &mut col_ptr,
                    &mut value_ptr
                ),
                SparseStatus::SUCCESS
            );
            assert_eq!((rows, columns, nnz), (1, cols, 1));
            assert_eq!((*row_ptr, *col_ptr, *value_ptr), (0, cols - 1, 7.0));
        }
    }

    include!("support/mkl_sparse_cases.rs");
    mod precision_s {
        use super::*;
        type Number = f32;
        use rivet_blas_sys::mkl::sparse_64::{
            mkl_sparse_s_add_64 as add, mkl_sparse_s_create_bsr_64 as create_bsr,
            mkl_sparse_s_create_coo_64 as create_coo, mkl_sparse_s_create_csc_64 as create_csc,
            mkl_sparse_s_create_csr_64 as create_csr, mkl_sparse_s_export_bsr_64 as export_bsr,
            mkl_sparse_s_export_coo_64 as export_coo, mkl_sparse_s_export_csc_64 as export_csc,
            mkl_sparse_s_export_csr_64 as export_csr, mkl_sparse_s_mm_64 as mm,
            mkl_sparse_s_mv_64 as mv, mkl_sparse_s_set_value_64 as set_value,
            mkl_sparse_s_spmmd_64 as spmmd, mkl_sparse_s_trsm_64 as trsm,
            mkl_sparse_s_trsv_64 as trsv, mkl_sparse_s_update_values_64 as update_values,
        };
        include!("support/mkl_sparse_precision_cases.rs");
    }

    mod precision_d {
        use super::*;
        type Number = f64;
        use rivet_blas_sys::mkl::sparse_64::{
            mkl_sparse_d_add_64 as add, mkl_sparse_d_create_bsr_64 as create_bsr,
            mkl_sparse_d_create_coo_64 as create_coo, mkl_sparse_d_create_csc_64 as create_csc,
            mkl_sparse_d_create_csr_64 as create_csr, mkl_sparse_d_export_bsr_64 as export_bsr,
            mkl_sparse_d_export_coo_64 as export_coo, mkl_sparse_d_export_csc_64 as export_csc,
            mkl_sparse_d_export_csr_64 as export_csr, mkl_sparse_d_mm_64 as mm,
            mkl_sparse_d_mv_64 as mv, mkl_sparse_d_set_value_64 as set_value,
            mkl_sparse_d_spmmd_64 as spmmd, mkl_sparse_d_trsm_64 as trsm,
            mkl_sparse_d_trsv_64 as trsv, mkl_sparse_d_update_values_64 as update_values,
        };
        include!("support/mkl_sparse_precision_cases.rs");
    }

    mod precision_c {
        use super::*;
        type Number = MklComplex8;
        use rivet_blas_sys::mkl::sparse_64::{
            mkl_sparse_c_add_64 as add, mkl_sparse_c_create_bsr_64 as create_bsr,
            mkl_sparse_c_create_coo_64 as create_coo, mkl_sparse_c_create_csc_64 as create_csc,
            mkl_sparse_c_create_csr_64 as create_csr, mkl_sparse_c_export_bsr_64 as export_bsr,
            mkl_sparse_c_export_coo_64 as export_coo, mkl_sparse_c_export_csc_64 as export_csc,
            mkl_sparse_c_export_csr_64 as export_csr, mkl_sparse_c_mm_64 as mm,
            mkl_sparse_c_mv_64 as mv, mkl_sparse_c_set_value_64 as set_value,
            mkl_sparse_c_spmmd_64 as spmmd, mkl_sparse_c_trsm_64 as trsm,
            mkl_sparse_c_trsv_64 as trsv, mkl_sparse_c_update_values_64 as update_values,
        };
        include!("support/mkl_sparse_precision_cases.rs");
    }

    mod precision_z {
        use super::*;
        type Number = MklComplex16;
        use rivet_blas_sys::mkl::sparse_64::{
            mkl_sparse_z_add_64 as add, mkl_sparse_z_create_bsr_64 as create_bsr,
            mkl_sparse_z_create_coo_64 as create_coo, mkl_sparse_z_create_csc_64 as create_csc,
            mkl_sparse_z_create_csr_64 as create_csr, mkl_sparse_z_export_bsr_64 as export_bsr,
            mkl_sparse_z_export_coo_64 as export_coo, mkl_sparse_z_export_csc_64 as export_csc,
            mkl_sparse_z_export_csr_64 as export_csr, mkl_sparse_z_mm_64 as mm,
            mkl_sparse_z_mv_64 as mv, mkl_sparse_z_set_value_64 as set_value,
            mkl_sparse_z_spmmd_64 as spmmd, mkl_sparse_z_trsm_64 as trsm,
            mkl_sparse_z_trsv_64 as trsv, mkl_sparse_z_update_values_64 as update_values,
        };
        include!("support/mkl_sparse_precision_cases.rs");
    }
}
