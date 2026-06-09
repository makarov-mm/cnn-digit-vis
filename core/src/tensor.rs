#[derive(Clone)]
pub struct Vol {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl Vol {
    pub fn zeros(c: usize, h: usize, w: usize) -> Vol {
        Vol { c, h, w, data: vec![0.0; c * h * w] }
    }

    #[inline]
    pub fn idx(&self, c: usize, y: usize, x: usize) -> usize {
        (c * self.h + y) * self.w + x
    }

    #[inline]
    pub fn get(&self, c: usize, y: usize, x: usize) -> f32 {
        self.data[self.idx(c, y, x)]
    }

    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
}

pub struct Rng { s: u64 }

impl Rng {
    pub fn new(seed: u64) -> Self { Self { s: (seed ^ 0x9E37_79B9_7F4A_7C15) | 1 } }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.s;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.s = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn unit(&mut self) -> f32 { (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 }
    pub fn below(&mut self, n: usize) -> usize { (self.next_u64() % n as u64) as usize }

    pub fn gaussian(&mut self) -> f32 {
        let u1 = self.unit().max(1e-7);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

pub fn argmax(v: &[f32]) -> usize {
    let mut bi = 0;
    for i in 1..v.len() {
        if v[i] > v[bi] { bi = i; }
    }
    bi
}
