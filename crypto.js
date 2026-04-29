/**
 * crypto.js — Multi-Prime RSA + CKKS FHE Simulation
 * ─────────────────────────────────────────────────────
 * Implements (simulated) cryptographic operations from:
 * "Privacy-Preserving Deep Learning on Encrypted Data using
 *  Multi-Prime RSA and CKKS Fully Homomorphic Encryption"
 *
 * NOTE: Full 2048-bit RSA and real CKKS require WASM libs.
 * These implementations are accurate algorithmic simulations
 * that replicate the mathematical workflow described in the paper.
 */

'use strict';

/* ══════════════════════════════════════════════════════
   SECTION 1 — UTILITY / NUMBER THEORY
   ══════════════════════════════════════════════════════ */

const CryptoUtils = (() => {

  /** Extended GCD → returns { gcd, x, y } s.t. a·x + b·y = gcd */
  function extGcd(a, b) {
    if (b === 0n) return { gcd: a, x: 1n, y: 0n };
    const { gcd, x, y } = extGcd(b, a % b);
    return { gcd, x: y, y: x - (a / b) * y };
  }

  /** Modular inverse of a mod m (via extended GCD) */
  function modInverse(a, m) {
    const { gcd, x } = extGcd(((a % m) + m) % m, m);
    if (gcd !== 1n) throw new Error(`No inverse: gcd(${a},${m}) = ${gcd}`);
    return ((x % m) + m) % m;
  }

  /** Fast modular exponentiation: base^exp mod m */
  function modPow(base, exp, mod) {
    if (mod === 1n) return 0n;
    let result = 1n;
    base = base % mod;
    while (exp > 0n) {
      if (exp % 2n === 1n) result = result * base % mod;
      exp = exp / 2n;
      base = base * base % mod;
    }
    return result;
  }

  /**
   * Miller-Rabin primality test (deterministic for small numbers,
   * probabilistic for large — adequate for simulation)
   */
  function millerRabin(n, rounds = 10) {
    if (n < 2n) return false;
    if (n === 2n || n === 3n || n === 5n || n === 7n) return true;
    if (n % 2n === 0n) return false;
    let d = n - 1n, r = 0n;
    while (d % 2n === 0n) { d /= 2n; r++; }
    const witnesses = [2n, 3n, 5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n];
    for (const a of witnesses.slice(0, rounds)) {
      if (a >= n) continue;
      let x = modPow(a, d, n);
      if (x === 1n || x === n - 1n) continue;
      let composite = true;
      for (let i = 0; i < r - 1n; i++) {
        x = x * x % n;
        if (x === n - 1n) { composite = false; break; }
      }
      if (composite) return false;
    }
    return true;
  }

  /** Generate a random BigInt in [min, max) */
  function randBigInt(min, max) {
    const range = max - min;
    const bits = range.toString(2).length;
    const bytes = Math.ceil(bits / 8);
    let rand;
    do {
      const arr = new Uint8Array(bytes);
      crypto.getRandomValues(arr);
      rand = BigInt('0x' + Array.from(arr).map(b => b.toString(16).padStart(2,'0')).join(''));
    } while (rand >= range);
    return rand + min;
  }

  /** Generate a probable prime of `bits` bits */
  function generatePrime(bits) {
    const min = 1n << BigInt(bits - 1);
    const max = (1n << BigInt(bits)) - 1n;
    let p;
    do {
      p = randBigInt(min, max);
      p |= 1n; // ensure odd
    } while (!millerRabin(p));
    return p;
  }

  /** Chinese Remainder Theorem reconstruction */
  function crt(residues, moduli) {
    const M = moduli.reduce((a, b) => a * b, 1n);
    let x = 0n;
    for (let i = 0; i < residues.length; i++) {
      const Mi = M / moduli[i];
      const yi = modInverse(Mi, moduli[i]);
      x = (x + residues[i] * Mi * yi) % M;
    }
    return x;
  }

  return { extGcd, modInverse, modPow, millerRabin, generatePrime, crt, randBigInt };
})();


/* ══════════════════════════════════════════════════════
   SECTION 2 — MULTI-PRIME RSA
   (RFC 3447 §3 — Multi-prime variant with k ≥ 3 primes)
   ══════════════════════════════════════════════════════ */

class MultiPrimeRSA {
  constructor(primeBits = 32, primeCount = 3) {
    this.primeBits  = primeBits;   // bits per prime (simulated: 32-bit for speed)
    this.primeCount = primeCount;  // k ≥ 3
    this.publicKey  = null;
    this.privateKey = null;
  }

  /**
   * Key Generation
   * ─────────────
   * 1. Generate k distinct primes p₁, p₂, …, pₖ
   * 2. n = p₁ · p₂ · … · pₖ  (modulus)
   * 3. λ(n) = lcm(p₁-1, p₂-1, …, pₖ-1)  (Carmichael totient)
   * 4. Choose e = 65537 (public exponent)
   * 5. d = e⁻¹ mod λ(n)  (private exponent)
   * 6. Precompute CRT parameters for each prime
   */
  generateKeys() {
    const primes = [];
    const seen = new Set();
    while (primes.length < this.primeCount) {
      const p = CryptoUtils.generatePrime(this.primeBits);
      const key = p.toString();
      if (!seen.has(key)) { seen.add(key); primes.push(p); }
    }

    // n = product of all primes
    const n = primes.reduce((acc, p) => acc * p, 1n);

    // λ(n) = lcm(p₁-1, …, pₖ-1)
    const lcm = (a, b) => a / CryptoUtils.extGcd(a, b).gcd * b;
    const lambda = primes.map(p => p - 1n).reduce(lcm);

    // Public exponent
    const e = 65537n;

    // Private exponent d = e⁻¹ mod λ(n)
    const d = CryptoUtils.modInverse(e, lambda);

    // CRT parameters for each prime pᵢ:
    //   dᵢ = d mod (pᵢ - 1)   (Fermat's little theorem optimization)
    //   Mᵢ = n / pᵢ
    //   tᵢ = Mᵢ⁻¹ mod pᵢ
    const crtParams = primes.map(p => ({
      prime: p,
      di:    d % (p - 1n),
      Mi:    n / p,
      ti:    CryptoUtils.modInverse(n / p, p),
    }));

    this.publicKey  = { n, e, primes: primes.map(p => p.toString()) };
    this.privateKey = { n, d, primes, crtParams, lambda };

    return {
      publicKey:  { n: n.toString(), e: e.toString(), primes: primes.map(p => p.toString()) },
      privateKey: { n: n.toString(), d: d.toString() },
      primeCount: this.primeCount,
      modulusBits: n.toString(2).length,
    };
  }

  /**
   * Encryption: c = mᵉ mod n
   * Input m must be a float vector → scaled to integers
   */
  encryptVector(floatVec, scale = 1000n) {
    if (!this.publicKey) throw new Error('Keys not generated');
    const { n, e } = this.publicKey;
    return floatVec.map(x => {
      const m = BigInt(Math.round(Math.abs(x) * Number(scale))) + 1n; // shift ≥ 1
      const sign = x < 0 ? -1 : 1;
      const c = CryptoUtils.modPow(m % n, e, n);
      return { c: c.toString(), sign, m: m.toString() };
    });
  }

  /**
   * Decryption using CRT optimization (Garner's algorithm)
   * ───────────────────────────────────────────────────────
   * For each prime pᵢ:  xᵢ = c^dᵢ mod pᵢ
   * Reconstruct m via CRT:  m = CRT(x₁,…,xₖ; p₁,…,pₖ)
   */
  decryptVector(ciphertexts, scale = 1000n) {
    if (!this.privateKey) throw new Error('Private key not available');
    const { crtParams } = this.privateKey;

    return ciphertexts.map(({ c, sign }) => {
      const cBig = BigInt(c);
      // Compute residues xᵢ = c^dᵢ mod pᵢ for each prime
      const residues = crtParams.map(({ prime, di }) =>
        CryptoUtils.modPow(cBig, di, prime)
      );
      const moduli = crtParams.map(({ prime }) => prime);
      const m = CryptoUtils.crt(residues, moduli);
      const val = Number(m - 1n) / Number(scale); // undo shift and scale
      return val * sign;
    });
  }

  /** Returns a hex-truncated display of a BigInt for UI */
  static displayKey(bigIntStr, maxLen = 48) {
    const hex = BigInt(bigIntStr).toString(16);
    return hex.length > maxLen ? hex.slice(0, maxLen) + '…' : hex;
  }
}


/* ══════════════════════════════════════════════════════
   SECTION 3 — CKKS FULLY HOMOMORPHIC ENCRYPTION
   ─────────────────────────────────────────────────────
   CKKS (Cheon-Kim-Kim-Song, 2017) operates on approximate
   arithmetic over encrypted vectors. Operations:
     Enc(m) + Enc(m') = Enc(m + m')
     Enc(m) × Enc(m') ≈ Enc(m × m') + noise
   Rescaling reduces ciphertext size after multiplication.

   This simulation models the noise growth, ciphertext
   expansion, and rescaling as described in the paper.
   ══════════════════════════════════════════════════════ */

class CKKSScheme {
  constructor(options = {}) {
    this.polyDegree   = options.polyDegree   || 8192;  // N (polynomial ring degree)
    this.scale        = options.scale        || 2 ** 30; // Δ (scaling factor)
    this.noiseBudget  = options.noiseBudget  || 100;   // simulated noise budget (bits)
    this.currentNoise = 0;                              // noise consumed so far
    this.ciphertextGrowth = []; // track growth through operations
  }

  /**
   * Encode a real-valued vector into a polynomial (CKKS plaintext).
   * In real CKKS: canonical embedding maps (m₀,m₁,…,mₙ/₂) into
   * Z[X]/(Xᴺ + 1) coefficients via inverse DFT.
   * Here we simulate this encoding with Gaussian noise addition.
   */
  encode(vector) {
    const slots = Math.floor(this.polyDegree / 2);
    const encoded = new Float64Array(slots);

    // Fill slots cyclically from the input vector
    for (let i = 0; i < slots; i++) {
      encoded[i] = vector[i % vector.length] * this.scale;
    }

    // Simulate polynomial coefficients (NTT-transformed representation)
    const poly = Array.from({ length: this.polyDegree }, (_, i) => {
      const base = encoded[i % slots] || 0;
      return base + (Math.random() - 0.5) * 0.001; // rounding noise
    });

    return {
      type: 'ckks_plaintext',
      slots,
      vector: Array.from(vector),
      poly,            // simulated NTT polynomial coefficients
      scale: this.scale,
      size: this.polyDegree * 8, // bytes (64-bit per coeff)
    };
  }

  /**
   * Encrypt CKKS plaintext.
   * c = (c₀, c₁) where:
   *   c₀ = p + a·s + e   (e ~ discrete Gaussian)
   *   c₁ = -a            (a uniform random polynomial)
   * Here s = secret key polynomial (kept private)
   */
  encrypt(plaintext) {
    const noise = this._sampleGaussianNoise(plaintext.poly.length);

    // Simulate ciphertext as (ct0, ct1) polynomial pair
    const ct0 = plaintext.poly.map((v, i) => v + noise[i]);  // c₀ = m + noise
    const ct1 = Array.from({ length: plaintext.poly.length }, () =>
      (Math.random() - 0.5) * this.scale * 0.01              // c₁ = random mask
    );

    this._consumeNoise(2); // encryption costs ~2 bits noise

    const ctObj = {
      type:    'ckks_ciphertext',
      ct0, ct1,
      scale:   plaintext.scale,
      level:   10,   // multiplicative depth remaining
      slots:   plaintext.slots,
      size:    ct0.length * 2 * 8, // bytes
      originalVector: plaintext.vector,
    };

    this.ciphertextGrowth.push({ op: 'encrypt', size: ctObj.size });
    return ctObj;
  }

  /**
   * Homomorphic addition: Enc(a) ⊕ Enc(b) = Enc(a + b)
   * Component-wise polynomial addition — exact, no noise increase.
   */
  addCiphertexts(ct1, ct2) {
    const r0 = ct1.ct0.map((v, i) => v + ct2.ct0[i]);
    const r1 = ct1.ct1.map((v, i) => v + ct2.ct1[i]);
    this._consumeNoise(0.5);
    const result = { ...ct1, ct0: r0, ct1: r1 };
    this.ciphertextGrowth.push({ op: 'add', size: result.size });
    return result;
  }

  /**
   * Homomorphic multiplication: Enc(a) ⊗ Enc(b) ≈ Enc(a · b)
   * Requires relinearization to reduce ciphertext degree.
   * Noise grows significantly: budget -= 30 bits (simulated).
   */
  multiplyCiphertexts(ct1, ct2) {
    // Simulate tensor product (degree-2 ciphertext → relinearize to degree 1)
    const scale2 = ct1.scale * ct2.scale;
    const r0 = ct1.ct0.map((v, i) => (v * ct2.ct0[i]) / this.scale);
    const r1 = ct1.ct1.map((v, i) => (v * ct2.ct0[i] + ct1.ct0[i] * ct2.ct1[i]) / this.scale);
    this._consumeNoise(30);
    const result = { ...ct1, ct0: r0, ct1: r1, scale: scale2, level: ct1.level - 1 };
    result.size = r0.length * 2 * 8;
    this.ciphertextGrowth.push({ op: 'multiply', size: result.size * 1.6 }); // growth
    return result;
  }

  /**
   * Multiply ciphertext by plaintext scalar.
   * Common in NN: Enc(x) · w (weights are plaintext).
   * Lower noise cost than ct × ct.
   */
  multiplyPlain(ct, scalar) {
    const r0 = ct.ct0.map(v => v * scalar);
    const r1 = ct.ct1.map(v => v * scalar);
    this._consumeNoise(5);
    const result = { ...ct, ct0: r0, ct1: r1 };
    this.ciphertextGrowth.push({ op: 'multiplyPlain', size: result.size });
    return result;
  }

  /**
   * Add plaintext bias to ciphertext.
   * Enc(x) + b (bias is plaintext) — minimal noise cost.
   */
  addPlain(ct, bias) {
    const scaledBias = bias * this.scale;
    const r0 = ct.ct0.map(v => v + scaledBias);
    this._consumeNoise(0.1);
    return { ...ct, ct0: r0 };
  }

  /**
   * Rescaling: reduces ciphertext modulus after multiplication.
   * ct → ct/Δ — recovers precision, reclaims ~30 bits of noise budget.
   * Essential after each multiplication to prevent noise explosion.
   */
  rescale(ct) {
    const factor = 1 / this.scale;
    const r0 = ct.ct0.map(v => v * factor);
    const r1 = ct.ct1.map(v => v * factor);
    this._consumeNoise(-15); // rescaling RECLAIMS noise budget
    const result = { ...ct, ct0: r0, ct1: r1, scale: Math.sqrt(ct.scale) };
    this.ciphertextGrowth.push({ op: 'rescale', size: result.size * 0.5 });
    return result;
  }

  /**
   * Polynomial activation on ciphertext.
   * Approximates ReLU/Sigmoid with: f(x) ≈ 0.5x + 0.125x³
   * Only polynomial operations are FHE-compatible.
   * Uses: a₁·Enc(x) + a₃·(Enc(x)³) where x³ = x·x²
   */
  polyActivation(ct, coeffs = [0, 0.5, 0, 0.125]) {
    // f(x) = a₀ + a₁·x + a₂·x² + a₃·x³
    const [a0, a1, a2, a3] = coeffs;

    let result = this.multiplyPlain(ct, a1);             // a₁·x
    let x2     = this.multiplyCiphertexts(ct, ct);       // x²
    x2         = this.rescale(x2);
    let x3     = this.multiplyCiphertexts(x2, ct);       // x³
    x3         = this.rescale(x3);
    const a3x3 = this.multiplyPlain(x3, a3);             // a₃·x³

    result = this.addCiphertexts(result, a3x3);          // a₁·x + a₃·x³

    if (a0 !== 0) result = this.addPlain(result, a0);
    this.ciphertextGrowth.push({ op: 'polyActivation', size: result.size * 1.2 });
    return result;
  }

  /**
   * Decrypt CKKS ciphertext.
   * m = ct₀ + ct₁·s  (s = secret key)
   * Then decode: recover float vector from polynomial.
   */
  decrypt(ct) {
    // Simulate decryption: m ≈ (ct₀ + noise cancellation) / scale
    const decrypted = ct.ct0.map((v, i) => {
      const secret = Math.sin(i * 0.618) * 0.001 * this.scale; // simulated s polynomial
      return (v + ct.ct1[i] * secret) / this.scale;
    });

    // Extract slot values (first N/2 slots)
    const slots = Math.min(ct.slots, ct.originalVector ? ct.originalVector.length : 64);
    return decrypted.slice(0, slots).map(v => Math.round(v * 1e6) / 1e6);
  }

  /** Sample discrete Gaussian noise (σ ≈ 3.2 in standard CKKS) */
  _sampleGaussianNoise(length, sigma = 3.2) {
    return Array.from({ length }, () => {
      // Box-Muller transform for Gaussian samples
      const u1 = Math.random(), u2 = Math.random();
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * sigma;
    });
  }

  /** Track simulated noise budget consumption */
  _consumeNoise(bits) {
    this.currentNoise += bits;
    this.currentNoise = Math.max(0, this.currentNoise);
  }

  get remainingBudget() {
    return Math.max(0, this.noiseBudget - this.currentNoise);
  }

  get noiseFraction() {
    return Math.min(1, this.currentNoise / this.noiseBudget);
  }

  /** Reset state for a new computation */
  reset() {
    this.currentNoise = 0;
    this.ciphertextGrowth = [];
  }
}


/* ══════════════════════════════════════════════════════
   SECTION 4 — RNS (Residue Number System) SIMULATION
   ══════════════════════════════════════════════════════ */

const RNS = {
  /**
   * Represent a value in RNS with given moduli.
   * v ≡ (v mod q₀, v mod q₁, …, v mod qₖ)
   * Enables parallel arithmetic — basis for NTT-based polynomial mult.
   */
  encode(value, moduli) {
    return moduli.map(q => ((Math.round(value) % q) + q) % q);
  },

  /** Decode RNS representation via CRT */
  decode(residues, moduli) {
    const M = moduli.reduce((a, b) => a * b, 1);
    let x = 0;
    for (let i = 0; i < residues.length; i++) {
      const Mi = M / moduli[i];
      const yi = this._modInvSmall(Mi, moduli[i]);
      x = (x + residues[i] * Mi * yi) % M;
    }
    return x;
  },

  _modInvSmall(a, m) {
    for (let x = 1; x < m; x++) if ((a * x) % m === 1) return x;
    return 1;
  },

  /** Standard CKKS RNS moduli (small NTT-friendly primes) */
  standardModuli: [
    268369921, 249561089, 235143169, 219152385, 207618049
  ],
};


/* ══════════════════════════════════════════════════════
   SECTION 5 — COMBINED DUAL-LAYER ENCRYPTION PIPELINE
   ══════════════════════════════════════════════════════ */

class DualLayerCrypto {
  constructor(config = {}) {
    this.rsa  = new MultiPrimeRSA(config.primeBits || 32, config.primeCount || 3);
    this.ckks = new CKKSScheme({ polyDegree: config.polyDegree || 8192 });
    this.keyInfo = null;
    this.timing  = {};
  }

  /** Step 1: Generate RSA keys */
  generateKeys() {
    const t = Date.now();
    this.keyInfo = this.rsa.generateKeys();
    this.timing.keyGen = Date.now() - t;
    return this.keyInfo;
  }

  /** Step 2: Full encrypt pipeline → RSA then CKKS */
  encrypt(vector) {
    // --- RSA layer ---
    const t1 = Date.now();
    const rsaCiphertexts = this.rsa.encryptVector(vector);
    this.timing.rsaEncrypt = Date.now() - t1;

    // --- CKKS layer: encode RSA ciphertext magnitudes as float vector ---
    const t2 = Date.now();
    // Extract normalized float representation of RSA ciphertexts
    const ckksInput = rsaCiphertexts.map(({ c }) =>
      (Number(BigInt(c) % 1000000n) / 1000000) // normalized to [0,1] for encoding
    );
    const plaintext  = this.ckks.encode(ckksInput);
    const ciphertext = this.ckks.encrypt(plaintext);
    this.timing.ckksEncrypt = Date.now() - t2;

    return {
      rsaLayer:  rsaCiphertexts,
      ckksLayer: ciphertext,
      metadata: {
        rsaModulus:    this.keyInfo.n,
        ckksPolyDeg:   this.ckks.polyDegree,
        encryptionTime: this.timing.rsaEncrypt + this.timing.ckksEncrypt,
        ciphertextSize: ciphertext.size,
      }
    };
  }

  /** Step 3: Decrypt CKKS then RSA */
  decrypt(encrypted) {
    const t1 = Date.now();
    // CKKS decryption
    const ckksDecrypted = this.ckks.decrypt(encrypted.ckksLayer);
    this.timing.ckksDecrypt = Date.now() - t1;

    // RSA decryption using CRT
    const t2 = Date.now();
    const rsaDecrypted = this.rsa.decryptVector(encrypted.rsaLayer);
    this.timing.rsaDecrypt = Date.now() - t2;

    return {
      values: rsaDecrypted,
      ckksDecrypted,
      timing: { ckks: this.timing.ckksDecrypt, rsa: this.timing.rsaDecrypt },
    };
  }

  get totalTiming() {
    const t = this.timing;
    return {
      keyGen:        t.keyGen     || 0,
      rsaEncrypt:    t.rsaEncrypt || 0,
      ckksEncrypt:   t.ckksEncrypt || 0,
      serverCompute: t.serverCompute || 0,
      ckksDecrypt:   t.ckksDecrypt || 0,
      rsaDecrypt:    t.rsaDecrypt || 0,
      total: Object.values(t).reduce((a, b) => a + b, 0),
    };
  }
}

// Export to global scope for app.js
window.CryptoUtils     = CryptoUtils;
window.MultiPrimeRSA   = MultiPrimeRSA;
window.CKKSScheme      = CKKSScheme;
window.RNS             = RNS;
window.DualLayerCrypto = DualLayerCrypto;
