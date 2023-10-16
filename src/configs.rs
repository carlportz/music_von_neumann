#[derive(Clone, PartialEq)]
pub enum OperationMode {
    Direct,
    Iterative,
    Auto,
}

impl ToString for OperationMode {
    fn to_string(&self) -> String {
        match self {
            OperationMode::Direct => String::from("direct"),
            OperationMode::Iterative => String::from("iterative"),
            OperationMode::Auto => String::from("auto"),
        }
    }
}
