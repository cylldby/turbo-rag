use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

pub fn random_unit_vec(dim: usize, rng: &mut SmallRng) -> Vec<f32> {
    let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.into_iter().map(|x| x / norm).collect()
}

pub fn make_rng(seed: u64) -> SmallRng {
    SmallRng::seed_from_u64(seed)
}
