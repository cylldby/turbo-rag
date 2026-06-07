pub use common::{EmbeddingBackend, Result, TurboError};

mod mock;
pub use mock::MockEmbedder;

// synthetic is always compiled; the EmbeddingBackend impl is feature-gated but
// random_unit_vec is a free function used by the bench command.
pub mod synthetic;
#[cfg(feature = "backend-synthetic")]
pub use synthetic::SyntheticEmbedder;
pub use synthetic::random_unit_vec;

#[cfg(feature = "backend-fastembed")]
mod fastembed_backend;
#[cfg(feature = "backend-fastembed")]
pub use fastembed_backend::FastEmbedBackend;

#[cfg(feature = "backend-openai-compat")]
mod openai_compat;
#[cfg(feature = "backend-openai-compat")]
pub use openai_compat::OpenAICompatBackend;
