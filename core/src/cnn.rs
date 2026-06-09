use crate::layers::{relu_inplace, softmax, Conv, Dense, MaxPool};
use crate::tensor::{argmax, Rng, Vol};
use std::fs::File;
use std::io::{Read, Write};

// 1x28x28 -> conv(8) -> pool -> conv(16) -> pool -> fc(64) -> fc(10)
pub struct Cnn {
    pub conv1: Conv,
    pub pool1: MaxPool,
    pub conv2: Conv,
    pub pool2: MaxPool,
    pub fc1: Dense,
    pub fc2: Dense,

    pub a_c1: Vol,
    pub a_p1: Vol,
    pub a_c2: Vol,
    pub a_p2: Vol,
    pub a_fc1: Vec<f32>,
    pub probs: Vec<f32>,
}

impl Cnn {
    pub fn new(seed: u64) -> Cnn {
        let mut rng = Rng::new(seed);
        Cnn {
            conv1: Conv::new(1, 8, 3, &mut rng),
            pool1: MaxPool::new(),
            conv2: Conv::new(8, 16, 3, &mut rng),
            pool2: MaxPool::new(),
            fc1: Dense::new(16 * 7 * 7, 64, &mut rng),
            fc2: Dense::new(64, 10, &mut rng),
            a_c1: Vol::zeros(0, 0, 0),
            a_p1: Vol::zeros(0, 0, 0),
            a_c2: Vol::zeros(0, 0, 0),
            a_p2: Vol::zeros(0, 0, 0),
            a_fc1: Vec::new(),
            probs: vec![0.1; 10],
        }
    }

    pub fn forward(&mut self, input: &Vol) -> &[f32] {
        let mut c1 = self.conv1.forward(input);
        relu_inplace(&mut c1.data);
        self.a_c1 = c1;
        self.a_p1 = self.pool1.forward(&self.a_c1);

        let mut c2 = self.conv2.forward(&self.a_p1);
        relu_inplace(&mut c2.data);
        self.a_c2 = c2;
        self.a_p2 = self.pool2.forward(&self.a_c2);

        let mut h = self.fc1.forward(&self.a_p2.data);
        relu_inplace(&mut h);
        self.a_fc1 = h;

        let logits = self.fc2.forward(&self.a_fc1);
        self.probs = softmax(&logits);
        &self.probs
    }

    pub(crate) fn backward(&mut self, label: usize) {
        let mut g: Vec<f32> = self.probs.clone();
        g[label] -= 1.0;

        let mut g_fc1 = self.fc2.backward(&g);
        for i in 0..g_fc1.len() {
            if self.a_fc1[i] <= 0.0 { g_fc1[i] = 0.0; }
        }

        let g_flat = self.fc1.backward(&g_fc1);
        let g_p2 = Vol { c: self.a_p2.c, h: self.a_p2.h, w: self.a_p2.w, data: g_flat };

        let mut g_c2 = self.pool2.backward(&g_p2);
        for i in 0..g_c2.data.len() {
            if self.a_c2.data[i] <= 0.0 { g_c2.data[i] = 0.0; }
        }

        let g_p1 = self.conv2.backward(&g_c2);

        let mut g_c1 = self.pool1.backward(&g_p1);
        for i in 0..g_c1.data.len() {
            if self.a_c1.data[i] <= 0.0 { g_c1.data[i] = 0.0; }
        }

        let _ = self.conv1.backward(&g_c1);
    }

    pub fn loss(&mut self, input: &Vol, label: usize) -> f32 {
        self.forward(input);
        -(self.probs[label].max(1e-12)).ln()
    }

    pub fn predict(&mut self, input: &Vol) -> usize {
        self.forward(input);
        argmax(&self.probs)
    }

    fn zero_grad(&mut self) {
        self.conv1.zero_grad();
        self.conv2.zero_grad();
        self.fc1.zero_grad();
        self.fc2.zero_grad();
    }

    fn step(&mut self, lr: f32, scale: f32) {
        self.conv1.step(lr, scale);
        self.conv2.step(lr, scale);
        self.fc1.step(lr, scale);
        self.fc2.step(lr, scale);
    }

    pub fn train_batch(&mut self, batch: &[(Vol, u8)], lr: f32) -> f32 {
        self.zero_grad();
        let mut loss = 0.0;
        for (x, y) in batch {
            let y = *y as usize;
            self.forward(x);
            loss += -(self.probs[y].max(1e-12)).ln();
            self.backward(y);
        }
        self.step(lr, 1.0 / batch.len() as f32);
        loss / batch.len() as f32
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut f = File::create(path)?;
        f.write_all(b"CNN1")?;
        for blob in [&self.conv1.w, &self.conv1.b, &self.conv2.w, &self.conv2.b,
                     &self.fc1.w, &self.fc1.b, &self.fc2.w, &self.fc2.b] {
            for v in blob.iter() { f.write_all(&v.to_le_bytes())?; }
        }
        Ok(())
    }

    pub fn try_load(&mut self, path: &str) -> bool {
        let Ok(mut f) = File::open(path) else { return false; };
        let mut magic = [0u8; 4];
        if f.read_exact(&mut magic).is_err() || &magic != b"CNN1" { return false; }
        let mut rest = Vec::new();
        if f.read_to_end(&mut rest).is_err() { return false; }

        let total = self.conv1.w.len() + self.conv1.b.len() + self.conv2.w.len() + self.conv2.b.len()
            + self.fc1.w.len() + self.fc1.b.len() + self.fc2.w.len() + self.fc2.b.len();
        if rest.len() != total * 4 { return false; }

        let mut off = 0;
        let read_into = |dst: &mut [f32], off: &mut usize| {
            for v in dst.iter_mut() {
                let bytes = [rest[*off], rest[*off + 1], rest[*off + 2], rest[*off + 3]];
                *v = f32::from_le_bytes(bytes);
                *off += 4;
            }
        };
        
        read_into(&mut self.conv1.w, &mut off);
        read_into(&mut self.conv1.b, &mut off);
        read_into(&mut self.conv2.w, &mut off);
        read_into(&mut self.conv2.b, &mut off);
        read_into(&mut self.fc1.w, &mut off);
        read_into(&mut self.fc1.b, &mut off);
        read_into(&mut self.fc2.w, &mut off);
        read_into(&mut self.fc2.b, &mut off);
        true
    }
}
