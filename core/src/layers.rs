use crate::tensor::{Rng, Vol};

pub struct Conv {
    pub in_c: usize,
    pub out_c: usize,
    pub k: usize,
    pub pad: usize,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub dw: Vec<f32>,
    pub db: Vec<f32>,
    input: Vol,
}

impl Conv {
    pub fn new(in_c: usize, out_c: usize, k: usize, rng: &mut Rng) -> Conv {
        let n = out_c * in_c * k * k;
        let scale = (2.0 / (in_c * k * k) as f32).sqrt();
        let w: Vec<f32> = (0..n).map(|_| rng.gaussian() * scale).collect();
        Conv {
            in_c,
            out_c,
            k,
            pad: k / 2,
            w,
            b: vec![0.0; out_c],
            dw: vec![0.0; n],
            db: vec![0.0; out_c],
            input: Vol::zeros(0, 0, 0),
        }
    }

    #[inline]
    fn widx(&self, oc: usize, ic: usize, ky: usize, kx: usize) -> usize {
        ((oc * self.in_c + ic) * self.k + ky) * self.k + kx
    }

    pub fn forward(&mut self, x: &Vol) -> Vol {
        self.input = x.clone();
        let (h, w) = (x.h, x.w);
        let mut out = Vol::zeros(self.out_c, h, w);
        let k = self.k;
        let pad = self.pad;
        for oc in 0..self.out_c {
            for oy in 0..h {
                for ox in 0..w {
                    let mut sum = self.b[oc];
                    for ic in 0..self.in_c {
                        for ky in 0..k {
                            let iy = oy + ky;
                            if iy < pad || iy - pad >= h { continue; }
                            let yy = iy - pad;
                            for kx in 0..k {
                                let ix = ox + kx;
                                if ix < pad || ix - pad >= w { continue; }
                                let xx = ix - pad;
                                sum += x.get(ic, yy, xx) * self.w[self.widx(oc, ic, ky, kx)];
                            }
                        }
                    }
                    let oi = out.idx(oc, oy, ox);
                    out.data[oi] = sum;
                }
            }
        }
        out
    }

    pub fn backward(&mut self, grad_out: &Vol) -> Vol {
        let (h, w) = (self.input.h, self.input.w);
        let mut grad_in = Vol::zeros(self.in_c, h, w);
        let k = self.k;
        let pad = self.pad;
        for oc in 0..self.out_c {
            for oy in 0..h {
                for ox in 0..w {
                    let g = grad_out.get(oc, oy, ox);
                    if g == 0.0 { continue; }
                    self.db[oc] += g;
                    for ic in 0..self.in_c {
                        for ky in 0..k {
                            let iy = oy + ky;
                            if iy < pad || iy - pad >= h { continue; }
                            let yy = iy - pad;
                            for kx in 0..k {
                                let ix = ox + kx;
                                if ix < pad || ix - pad >= w { continue; }
                                let xx = ix - pad;
                                let wi = self.widx(oc, ic, ky, kx);
                                self.dw[wi] += g * self.input.get(ic, yy, xx);
                                let gi = grad_in.idx(ic, yy, xx);
                                grad_in.data[gi] += g * self.w[wi];
                            }
                        }
                    }
                }
            }
        }
        grad_in
    }

    pub fn zero_grad(&mut self) {
        self.dw.iter_mut().for_each(|v| *v = 0.0);
        self.db.iter_mut().for_each(|v| *v = 0.0);
    }

    pub fn step(&mut self, lr: f32, scale: f32) {
        for i in 0..self.w.len() { self.w[i] -= lr * self.dw[i] * scale; }
        for i in 0..self.b.len() { self.b[i] -= lr * self.db[i] * scale; }
    }
}

pub struct MaxPool {
    argmax: Vec<usize>,
    in_c: usize,
    in_h: usize,
    in_w: usize,
}

impl MaxPool {
    pub fn new() -> MaxPool { MaxPool { argmax: Vec::new(), in_c: 0, in_h: 0, in_w: 0 } }

    pub fn forward(&mut self, x: &Vol) -> Vol {
        let (oh, ow) = (x.h / 2, x.w / 2);
        self.in_c = x.c;
        self.in_h = x.h;
        self.in_w = x.w;
        self.argmax = vec![0; x.c * oh * ow];
        let mut out = Vol::zeros(x.c, oh, ow);
        for c in 0..x.c {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut best = f32::MIN;
                    let mut bi = 0;
                    for j in 0..4 {
                        let yy = oy * 2 + j / 2;
                        let xx = ox * 2 + j % 2;
                        let v = x.get(c, yy, xx);
                        if v > best { best = v; bi = j; }
                    }
                    let oi = out.idx(c, oy, ox);
                    out.data[oi] = best;
                    self.argmax[oi] = bi;
                }
            }
        }
        out
    }

    pub fn backward(&self, grad_out: &Vol) -> Vol {
        let mut grad_in = Vol::zeros(self.in_c, self.in_h, self.in_w);
        let (oh, ow) = (grad_out.h, grad_out.w);
        for c in 0..grad_out.c {
            for oy in 0..oh {
                for ox in 0..ow {
                    let oi = grad_out.idx(c, oy, ox);
                    let j = self.argmax[oi];
                    let yy = oy * 2 + j / 2;
                    let xx = ox * 2 + j % 2;
                    let gi = grad_in.idx(c, yy, xx);
                    grad_in.data[gi] = grad_out.data[oi];
                }
            }
        }
        grad_in
    }
}

pub struct Dense {
    pub n_in: usize,
    pub n_out: usize,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub dw: Vec<f32>,
    pub db: Vec<f32>,
    input: Vec<f32>,
}

impl Dense {
    pub fn new(n_in: usize, n_out: usize, rng: &mut Rng) -> Dense {
        let scale = (2.0 / n_in as f32).sqrt();
        let w: Vec<f32> = (0..n_in * n_out).map(|_| rng.gaussian() * scale).collect();
        Dense {
            n_in,
            n_out,
            w,
            b: vec![0.0; n_out],
            dw: vec![0.0; n_in * n_out],
            db: vec![0.0; n_out],
            input: Vec::new(),
        }
    }

    pub fn forward(&mut self, x: &[f32]) -> Vec<f32> {
        self.input = x.to_vec();
        let mut out = vec![0.0f32; self.n_out];
        for j in 0..self.n_out {
            let mut sum = self.b[j];
            let base = j * self.n_in;
            for i in 0..self.n_in {
                sum += x[i] * self.w[base + i];
            }
            out[j] = sum;
        }
        out
    }

    pub fn backward(&mut self, grad_out: &[f32]) -> Vec<f32> {
        let mut grad_in = vec![0.0f32; self.n_in];
        for j in 0..self.n_out {
            let g = grad_out[j];
            self.db[j] += g;
            let base = j * self.n_in;
            for i in 0..self.n_in {
                self.dw[base + i] += g * self.input[i];
                grad_in[i] += g * self.w[base + i];
            }
        }
        grad_in
    }

    pub fn zero_grad(&mut self) {
        self.dw.iter_mut().for_each(|v| *v = 0.0);
        self.db.iter_mut().for_each(|v| *v = 0.0);
    }

    pub fn step(&mut self, lr: f32, scale: f32) {
        for i in 0..self.w.len() { self.w[i] -= lr * self.dw[i] * scale; }
        for i in 0..self.b.len() { self.b[i] -= lr * self.db[i] * scale; }
    }
}

pub fn relu_inplace(v: &mut [f32]) {
    for x in v.iter_mut() {
        if *x < 0.0 { *x = 0.0; }
    }
}

pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().cloned().fold(f32::MIN, f32::max);
    let mut e: Vec<f32> = logits.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    for v in e.iter_mut() { *v /= s; }
    e
}
