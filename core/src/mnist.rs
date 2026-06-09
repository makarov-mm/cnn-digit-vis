use crate::tensor::Vol;
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::Read;
use std::path::Path;

fn read_gz(path: &Path) -> std::io::Result<Vec<u8>> {
    let f = File::open(path)?;
    let mut gz = GzDecoder::new(f);
    let mut buf = Vec::new();
    gz.read_to_end(&mut buf)?;
    Ok(buf)
}

fn be32(b: &[u8], o: usize) -> usize {
    ((b[o] as usize) << 24) | ((b[o + 1] as usize) << 16) | ((b[o + 2] as usize) << 8) | b[o + 3] as usize
}

pub fn load(dir: &str, test: bool) -> std::io::Result<Vec<(Vol, u8)>> {
    let d = Path::new(dir);
    let (img, lbl) = if test {
        ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
    } else {
        ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
    };

    let images = read_gz(&d.join(img))?;
    let labels = read_gz(&d.join(lbl))?;

    let n = be32(&images, 4);
    let rows = be32(&images, 8);
    let cols = be32(&images, 12);
    let stride = rows * cols;

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = 16 + i * stride;
        let data: Vec<f32> = images[off..off + stride].iter().map(|&p| p as f32 / 255.0).collect();
        out.push((Vol { c: 1, h: rows, w: cols, data }, labels[8 + i]));
    }
    Ok(out)
}

pub fn preprocess(src: &[f32], sw: usize, sh: usize) -> Vol {
    let thr = 0.1;
    let (mut minx, mut miny, mut maxx, mut maxy) = (sw, sh, 0usize, 0usize);
    let mut any = false;
    for y in 0..sh {
        for x in 0..sw {
            if src[y * sw + x] > thr {
                any = true;
                minx = minx.min(x);
                miny = miny.min(y);
                maxx = maxx.max(x);
                maxy = maxy.max(y);
            }
        }
    }
    if !any {
        return Vol::zeros(1, 28, 28);
    }

    let bw = maxx - minx + 1;
    let bh = maxy - miny + 1;
    let scale = 20.0 / bw.max(bh) as f32;
    let tw = ((bw as f32 * scale).round() as usize).max(1);
    let th = ((bh as f32 * scale).round() as usize).max(1);
    let mut temp = vec![0.0f32; tw * th];

    for ty in 0..th {
        for tx in 0..tw {
            let x0 = minx as f32 + tx as f32 / tw as f32 * bw as f32;
            let x1 = minx as f32 + (tx + 1) as f32 / tw as f32 * bw as f32;
            let y0 = miny as f32 + ty as f32 / th as f32 * bh as f32;
            let y1 = miny as f32 + (ty + 1) as f32 / th as f32 * bh as f32;
            let (mut sum, mut cnt) = (0.0f32, 0.0f32);
            for yy in y0.floor() as usize..=(y1.ceil() as usize).min(sh - 1) {
                for xx in x0.floor() as usize..=(x1.ceil() as usize).min(sw - 1) {
                    sum += src[yy * sw + xx];
                    cnt += 1.0;
                }
            }
            temp[ty * tw + tx] = if cnt > 0.0 { sum / cnt } else { 0.0 };
        }
    }

    let (mut mx, mut my, mut m) = (0.0f32, 0.0f32, 0.0f32);
    for ty in 0..th {
        for tx in 0..tw {
            let v = temp[ty * tw + tx];
            mx += v * tx as f32;
            my += v * ty as f32;
            m += v;
        }
    }
    if m < 1e-6 {
        return Vol::zeros(1, 28, 28);
    }
    let (cx, cy) = (mx / m, my / m);

    let ox = (14.0 - cx).round() as isize;
    let oy = (14.0 - cy).round() as isize;

    let mut out = Vol::zeros(1, 28, 28);
    for ty in 0..th {
        for tx in 0..tw {
            let dx = tx as isize + ox;
            let dy = ty as isize + oy;
            if dx >= 0 && dx < 28 && dy >= 0 && dy < 28 {
                out.data[dy as usize * 28 + dx as usize] = temp[ty * tw + tx];
            }
        }
    }
    out
}
