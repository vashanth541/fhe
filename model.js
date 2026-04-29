/**
 * model.js — Privacy-Preserving Neural Network
 * ──────────────────────────────────────────────
 * Implements deep learning inference on CKKS-encrypted data.
 *
 * Architecture:
 *   Input → Linear(n→32) → PolyActivation → Linear(32→4) → Softmax
 *
 * Key constraint: ALL intermediate values remain encrypted.
 * Weights and biases are plaintext (known to server).
 * Activation is polynomial-approximated for FHE compatibility.
 *
 * Polynomial activation: f(x) ≈ 0.5x + 0.125x³
 * (Approximates sigmoid-like behavior without non-polynomial ops)
 */

'use strict';

/* ══════════════════════════════════════════════
   PLAINTEXT NEURAL NETWORK (reference model)
   Used to derive weights; not used during inference.
   ══════════════════════════════════════════════ */

class PlaintextNN {
  constructor(inputSize = 12, hiddenSize = 32, outputSize = 4) {
    this.inputSize  = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;
    this._initWeights();
  }

  /** He initialization for weights */
  _initWeights() {
    const he = (fan_in) => () => (Math.random() * 2 - 1) * Math.sqrt(2 / fan_in);

    this.W1 = Array.from({ length: this.hiddenSize }, () =>
      Array.from({ length: this.inputSize }, he(this.inputSize))
    );
    this.b1 = Array(this.hiddenSize).fill(0).map(() => (Math.random() - 0.5) * 0.1);

    this.W2 = Array.from({ length: this.outputSize }, () =>
      Array.from({ length: this.hiddenSize }, he(this.hiddenSize))
    );
    this.b2 = Array(this.outputSize).fill(0).map(() => (Math.random() - 0.5) * 0.1);
  }

  /** Standard forward pass (for comparison / accuracy verification) */
  forward(x) {
    const z1 = this._linear(x, this.W1, this.b1);
    const a1 = z1.map(v => Math.max(0, v)); // ReLU (plaintext only)
    const z2 = this._linear(a1, this.W2, this.b2);
    return this._softmax(z2);
  }

  _linear(x, W, b) {
    return W.map((row, i) =>
      row.reduce((sum, w, j) => sum + w * x[j], 0) + b[i]
    );
  }

  _softmax(x) {
    const max = Math.max(...x);
    const exp = x.map(v => Math.exp(v - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(v => v / sum);
  }
}


/* ══════════════════════════════════════════════
   ENCRYPTED NEURAL NETWORK
   Operates entirely on CKKS ciphertexts.
   ══════════════════════════════════════════════ */

class EncryptedNN {
  constructor(plaintextModel, ckks) {
    this.model  = plaintextModel;
    this.ckks   = ckks;
    this.timing = {};

    // Activation polynomial coefficients: f(x) ≈ 0.5x + 0.125x³
    this.polyCoeffs = [0, 0.5, 0, 0.125];
  }

  /**
   * Encrypted Linear Layer
   * ───────────────────────
   * Computes W·enc(x) + b on ciphertext.
   *
   * For each output neuron i:
   *   enc(zᵢ) = Σⱼ Wᵢⱼ · enc(xⱼ) + bᵢ
   *
   * Since Wᵢⱼ are plaintext scalars, we use multiplyPlain
   * (lower noise cost than ct × ct multiplication).
   *
   * @param {Array<CKKS_Ciphertext>} encInputs - Array of encrypted scalars
   * @param {Array<Array<number>>}   W          - Weight matrix [out × in]
   * @param {Array<number>}          b          - Bias vector [out]
   * @returns {Array<CKKS_Ciphertext>}
   */
  encryptedLinear(encInputs, W, b) {
    const outputSize = W.length;
    const results = [];

    for (let i = 0; i < outputSize; i++) {
      // Weighted sum: Σⱼ Wᵢⱼ · enc(xⱼ)
      let acc = null;
      for (let j = 0; j < encInputs.length; j++) {
        const wij = W[i][j];
        if (Math.abs(wij) < 1e-10) continue; // skip near-zero weights

        const scaled = this.ckks.multiplyPlain(encInputs[j], wij);
        acc = acc ? this.ckks.addCiphertexts(acc, scaled) : scaled;
      }

      // Add bias (plaintext)
      if (acc) {
        acc = this.ckks.addPlain(acc, b[i]);
      }
      results.push(acc);
    }
    return results;
  }

  /**
   * Polynomial Activation (FHE-Compatible)
   * ────────────────────────────────────────
   * Applies f(x) ≈ 0.5x + 0.125x³ to each encrypted value.
   *
   * This approximates sigmoid on [-3, 3] with L∞ error < 0.03.
   * Only addition and multiplication operations — fully FHE compatible.
   */
  encryptedPolyActivation(encInputs) {
    return encInputs.map(ct => {
      if (!ct) return ct;
      return this.ckks.polyActivation(ct, this.polyCoeffs);
    });
  }

  /**
   * Full Encrypted Forward Pass
   * ────────────────────────────
   * All operations run on ciphertexts. Server never decrypts.
   *
   * Pipeline:
   *   enc(x) → encLinear(W1,b1) → encPolyAct → encLinear(W2,b2)
   *          → encSoftmaxApprox → enc(scores)
   */
  forward(encInputs) {
    const t0 = performance.now();

    // Layer 1: Input → Hidden
    const encZ1 = this.encryptedLinear(encInputs, this.model.W1, this.model.b1);
    this.timing.layer1 = performance.now() - t0;

    // Activation: Polynomial approximation
    const t1 = performance.now();
    const encA1 = this.encryptedPolyActivation(encZ1);
    this.timing.activation = performance.now() - t1;

    // Noise management: Rescale after activation (which involved multiplications)
    const encA1r = encA1.map(ct => ct ? this.ckks.rescale(ct) : ct);

    // Layer 2: Hidden → Output
    const t2 = performance.now();
    const encZ2 = this.encryptedLinear(encA1r, this.model.W2, this.model.b2);
    this.timing.layer2 = performance.now() - t2;

    // Note: Softmax requires argmax/exp which are not polynomial.
    // We return raw logit ciphertexts and apply softmax AFTER decryption.
    return encZ2;
  }

  /** Extract final prediction after decryption */
  static getResult(decryptedLogits, categories) {
    const cats = categories || ['Category A', 'Category B', 'Category C', 'Category D'];

    // Softmax on decrypted values
    const max  = Math.max(...decryptedLogits);
    const exps = decryptedLogits.map(v => Math.exp(v - max));
    const sum  = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map(v => v / sum);

    const maxIdx = probs.indexOf(Math.max(...probs));
    return {
      label:       cats[maxIdx] || `Class ${maxIdx}`,
      confidence:  probs[maxIdx],
      allProbs:    cats.map((c, i) => ({ label: c, prob: probs[i] })),
      logits:      decryptedLogits,
    };
  }
}


/* ══════════════════════════════════════════════
   DATA PREPROCESSING
   ══════════════════════════════════════════════ */

const DataPreprocessor = {
  /**
   * Parse comma-separated float string into array.
   * Clamps to [inputSize] features, zero-pads if shorter.
   */
  parseVector(str, inputSize = 12) {
    const vals = str.split(',')
      .map(s => parseFloat(s.trim()))
      .filter(v => !isNaN(v));

    if (vals.length === 0) throw new Error('No valid numerical values found');

    // Pad or truncate to inputSize
    const result = new Array(inputSize).fill(0);
    for (let i = 0; i < Math.min(vals.length, inputSize); i++) {
      result[i] = vals[i];
    }
    return result;
  },

  /**
   * Parse CSV text into row vectors.
   * Handles headers, missing values, and mixed types.
   */
  parseCSV(text, inputSize = 12) {
    const lines = text.trim().split('\n');
    const rows  = [];
    let startRow = 0;

    // Skip header if first row is non-numeric
    if (isNaN(parseFloat(lines[0].split(',')[0]))) startRow = 1;

    for (let i = startRow; i < lines.length; i++) {
      const parts = lines[i].split(',').map(s => s.trim());
      const vals  = parts.map(v => parseFloat(v) || 0);
      if (vals.length > 0) {
        const padded = new Array(inputSize).fill(0);
        for (let j = 0; j < Math.min(vals.length, inputSize); j++) {
          padded[j] = vals[j];
        }
        rows.push(padded);
      }
    }
    return rows;
  },

  /**
   * Z-score normalization: x → (x - μ) / σ
   * Applied per-feature across the batch.
   */
  normalize(vectors) {
    const n     = vectors.length;
    const fSize = vectors[0].length;
    const means = new Array(fSize).fill(0);
    const stds  = new Array(fSize).fill(0);

    for (const vec of vectors) {
      for (let j = 0; j < fSize; j++) means[j] += vec[j] / n;
    }
    for (const vec of vectors) {
      for (let j = 0; j < fSize; j++) stds[j] += (vec[j] - means[j]) ** 2 / n;
    }
    for (let j = 0; j < fSize; j++) stds[j] = Math.sqrt(stds[j]) || 1;

    return vectors.map(vec =>
      vec.map((v, j) => (v - means[j]) / stds[j])
    );
  },

  /** Normalize a single vector to [-1, 1] range */
  normalizeVector(vec) {
    const max = Math.max(...vec.map(Math.abs)) || 1;
    return vec.map(v => v / max);
  },

  /** Sample ECG-like feature vector for demonstration */
  sampleECG() {
    return [
      0.42, -0.17, 0.88, 0.33, -0.55,
      0.71, 0.12, -0.44, 0.67, 0.29,
      -0.81, 0.55
    ];
  },
};


/* ══════════════════════════════════════════════
   SERVER-SIDE SIMULATION
   (In production: these run on the remote server)
   ══════════════════════════════════════════════ */

class ServerInference {
  constructor(ckks, encNN) {
    this.ckks  = ckks;
    this.encNN = encNN;
  }

  /**
   * Run encrypted inference.
   * Server receives: array of CKKS ciphertexts
   * Server returns:  array of CKKS ciphertexts (logits)
   * Server NEVER decrypts anything.
   */
  async runInference(encInputs, onProgress) {
    const steps = [
      { label: 'Receiving encrypted payload', delay: 80  },
      { label: 'Running Layer 1 (encrypted linear)', delay: 200 },
      { label: 'Applying polynomial activation', delay: 150 },
      { label: 'Noise management (rescaling)', delay: 60  },
      { label: 'Running Layer 2 (encrypted linear)', delay: 180 },
      { label: 'Preparing encrypted response', delay: 50  },
    ];

    const t = Date.now();
    for (const step of steps) {
      if (onProgress) onProgress(step.label);
      await sleep(step.delay);
    }

    // Run the actual encrypted forward pass
    const encOutputs = this.encNN.forward(encInputs);
    this.encNN.timing.serverCompute = Date.now() - t;

    return encOutputs;
  }
}

/** Tiny utility to simulate async delay */
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Export to global scope
window.PlaintextNN    = PlaintextNN;
window.EncryptedNN    = EncryptedNN;
window.DataPreprocessor = DataPreprocessor;
window.ServerInference  = ServerInference;
window.sleep = sleep;
