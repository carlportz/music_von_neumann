use crate::configs::OperationMode;
use crate::types::Float;

// main
pub const DEFAULT_NPOINT: usize = 128 * 128;
pub const DEFAULT_FREQ_MAX: Float = 4000.0;
pub const DEFAULT_OPERATION_MODE: OperationMode = OperationMode::Auto;
pub const DEFAULT_DIRECT_CUTOFF_NPOINT: usize = 64 * 64;

// bicgstab
pub const DEFAULT_BICGSTAB_TOL: Float = 1e-8;
