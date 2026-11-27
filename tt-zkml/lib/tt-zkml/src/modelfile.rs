//! The on-disk format of a model and of an input.
//!
//! A model is a directory:
//!
//! ```text
//! model/
//!   model.json     {"dtype": "bf16", "layers": [{"in": 64, "out": 128, "relu": true, "weights": "w0.f32"}, ...]}
//!   w0.f32 ...     the W of a layer, [in, out] row-major, little-endian f32
//! ```
//!
//! An input is a little-endian f32 file, `[batch, in]` row-major; `batch` follows from its size.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::device::{Dtype, Layer};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LayerEntry {
    #[serde(rename = "in")]
    pub in_features: u32,
    #[serde(rename = "out")]
    pub out_features: u32,
    #[serde(default)]
    pub relu: bool,
    /// File name of the weights, relative to the model directory.
    pub weights: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Manifest {
    pub dtype: String,
    pub layers: Vec<LayerEntry>,
}

/// A model read from disk.
#[derive(Debug, Clone)]
pub struct ModelFile {
    pub dtype: Dtype,
    pub layers: Vec<Layer>,
    pub weights: Vec<Vec<f32>>,
}

pub fn parse_dtype(s: &str) -> Result<Dtype, String> {
    match s {
        "f32" | "float32" => Ok(Dtype::F32),
        "bf16" | "bfloat16" => Ok(Dtype::Bf16),
        other => Err(format!("unknown dtype '{other}' (use f32 or bf16)")),
    }
}

pub fn dtype_name(d: Dtype) -> &'static str {
    match d {
        Dtype::F32 => "f32",
        Dtype::Bf16 => "bf16",
    }
}

pub fn read_f32(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{}: the size is not a multiple of 4 bytes", path.display()));
    }
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

pub fn write_f32(path: &Path, values: &[f32]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// Reads `model.json` and the weights of the model directory `dir`; checks the sizes.
pub fn load_model(dir: &Path) -> Result<ModelFile, String> {
    let manifest_path = dir.join("model.json");
    let text = fs::read_to_string(&manifest_path).map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let m: Manifest = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let dtype = parse_dtype(&m.dtype)?;
    let mut layers = Vec::new();
    let mut weights = Vec::new();
    for (i, l) in m.layers.iter().enumerate() {
        if l.in_features == 0 || l.out_features == 0 {
            return Err(format!("layer {i}: a layer needs at least one input and one output feature"));
        }
        // the weights are a file of the model directory, not a path out of it
        if l.weights.is_empty() || l.weights.contains(['/', '\\']) || l.weights == ".." {
            return Err(format!("layer {i}: the weights must be a file name in the model directory, not '{}'", l.weights));
        }
        let w = read_f32(&dir.join(&l.weights))?;
        let want = l.in_features as usize * l.out_features as usize;
        if w.len() != want {
            return Err(format!("layer {i}: {} has {} values, expected {} ({} x {})", l.weights, w.len(), want, l.in_features, l.out_features));
        }
        layers.push(Layer { in_features: l.in_features, out_features: l.out_features, relu: l.relu });
        weights.push(w);
    }
    if layers.is_empty() {
        return Err("the model has no layers".into());
    }
    Ok(ModelFile { dtype, layers, weights })
}

/// Writes a model directory (`model.json` and the weights).
pub fn save_model(dir: &Path, model: &ModelFile) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut entries = Vec::new();
    for (i, (l, w)) in model.layers.iter().zip(&model.weights).enumerate() {
        let name = format!("w{i}.f32");
        write_f32(&dir.join(&name), w)?;
        entries.push(LayerEntry { in_features: l.in_features, out_features: l.out_features, relu: l.relu, weights: name });
    }
    let manifest = Manifest { dtype: dtype_name(model.dtype).to_string(), layers: entries };
    let text = serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?;
    fs::write(dir.join("model.json"), text).map_err(|e| format!("model.json: {e}"))
}
