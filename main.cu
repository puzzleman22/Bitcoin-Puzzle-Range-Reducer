// Bitcoin Private Key Range Scanner - Corrected Implementation
// Based on working example with proper hash160 and search strategy

#include "secp256k1.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <string>
#define BLOCKS 256
#define THREADS_PER_BLOCK 256
#define CHECK_INTERVAL 2000

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


// Global device memory for results
__device__ volatile int g_found = 0;
__device__ char g_found_hex[65] = {0};
__device__ char g_found_hash160[41] = {0};

// Optimized SHA256 implementation
__device__ inline uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

__device__ void sha256(const uint8_t* data, int len, uint8_t hash[32]) {
    const uint32_t K[] = {
        0x428a2f98ul,0x71374491ul,0xb5c0fbcful,0xe9b5dba5ul,
        0x3956c25bul,0x59f111f1ul,0x923f82a4ul,0xab1c5ed5ul,
        0xd807aa98ul,0x12835b01ul,0x243185beul,0x550c7dc3ul,
        0x72be5d74ul,0x80deb1feul,0x9bdc06a7ul,0xc19bf174ul,
        0xe49b69c1ul,0xefbe4786ul,0x0fc19dc6ul,0x240ca1ccul,
        0x2de92c6ful,0x4a7484aaul,0x5cb0a9dcul,0x76f988daul,
        0x983e5152ul,0xa831c66dul,0xb00327c8ul,0xbf597fc7ul,
        0xc6e00bf3ul,0xd5a79147ul,0x06ca6351ul,0x14292967ul,
        0x27b70a85ul,0x2e1b2138ul,0x4d2c6dfcul,0x53380d13ul,
        0x650a7354ul,0x766a0abbul,0x81c2c92eul,0x92722c85ul,
        0xa2bfe8a1ul,0xa81a664bul,0xc24b8b70ul,0xc76c51a3ul,
        0xd192e819ul,0xd6990624ul,0xf40e3585ul,0x106aa070ul,
        0x19a4c116ul,0x1e376c08ul,0x2748774cul,0x34b0bcb5ul,
        0x391c0cb3ul,0x4ed8aa4aul,0x5b9cca4ful,0x682e6ff3ul,
        0x748f82eeul,0x78a5636ful,0x84c87814ul,0x8cc70208ul,
        0x90befffaul,0xa4506cebul,0xbef9a3f7ul,0xc67178f2ul
    };

    uint32_t h[8] = {
        0x6a09e667ul, 0xbb67ae85ul, 0x3c6ef372ul, 0xa54ff53aul,
        0x510e527ful, 0x9b05688cul, 0x1f83d9abul, 0x5be0cd19ul
    };

    uint8_t full[64] = {0};
    
    #pragma unroll
    for (int i = 0; i < len; ++i) full[i] = data[i];
    full[len] = 0x80;
    
    uint64_t bit_len = (uint64_t)len * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        full[63 - i] = bit_len >> (8 * i);
    }

    uint32_t w[64];
    
    #pragma unroll 16
    for (int i = 0; i < 16; ++i) {
        w[i] = (full[4 * i] << 24) | (full[4 * i + 1] << 16) |
               (full[4 * i + 2] << 8) | full[4 * i + 3];
    }
    
    #pragma unroll
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hval = h[7];

    #pragma unroll 8
    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = hval + S1 + ch + K[i] + w[i];
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        hval = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hval;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        hash[4 * i + 0] = (h[i] >> 24) & 0xFF;
        hash[4 * i + 1] = (h[i] >> 16) & 0xFF;
        hash[4 * i + 2] = (h[i] >> 8) & 0xFF;
        hash[4 * i + 3] = (h[i] >> 0) & 0xFF;
    }
}

// Proper RIPEMD160 implementation
__device__ void ripemd160(const uint8_t* msg, uint8_t* out) {
    const uint32_t K1[5] = {0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E};
    const uint32_t K2[5] = {0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000};
    
    const int ZL[80] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
        3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
        1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
        4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
    };
    
    const int ZR[80] = {
        5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
        6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
        15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
        8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
        12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
    };
    
    const int SL[80] = {
        11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
        7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
        11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
        11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
        9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6
    };
    
    const int SR[80] = {
        8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
        9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
        9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
        15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
        8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11
    };
    
    uint32_t h0 = 0x67452301;
    uint32_t h1 = 0xEFCDAB89;
    uint32_t h2 = 0x98BADCFE;
    uint32_t h3 = 0x10325476;
    uint32_t h4 = 0xC3D2E1F0;
    
    uint8_t buffer[64];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        buffer[i] = msg[i];
    }
    
    buffer[32] = 0x80;
    #pragma unroll
    for (int i = 33; i < 56; i++) {
        buffer[i] = 0x00;
    }
    
    uint64_t bitlen = 256;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        buffer[56 + i] = (bitlen >> (i * 8)) & 0xFF;
    }
    
    uint32_t X[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        X[i] = ((uint32_t)buffer[i*4]) | 
               ((uint32_t)buffer[i*4 + 1] << 8) | 
               ((uint32_t)buffer[i*4 + 2] << 16) | 
               ((uint32_t)buffer[i*4 + 3] << 24);
    }
    
    uint32_t AL = h0, BL = h1, CL = h2, DL = h3, EL = h4;
    uint32_t AR = h0, BR = h1, CR = h2, DR = h3, ER = h4;
    
    #pragma unroll 10
    for (int j = 0; j < 80; j++) {
        uint32_t T;
        
        if (j < 16) {
            T = AL + (BL ^ CL ^ DL) + X[ZL[j]] + K1[0];
        } else if (j < 32) {
            T = AL + ((BL & CL) | (~BL & DL)) + X[ZL[j]] + K1[1];
        } else if (j < 48) {
            T = AL + ((BL | ~CL) ^ DL) + X[ZL[j]] + K1[2];
        } else if (j < 64) {
            T = AL + ((BL & DL) | (CL & ~DL)) + X[ZL[j]] + K1[3];
        } else {
            T = AL + (BL ^ (CL | ~DL)) + X[ZL[j]] + K1[4];
        }
        T = ((T << SL[j]) | (T >> (32 - SL[j]))) + EL;
        AL = EL; EL = DL; DL = (CL << 10) | (CL >> 22); CL = BL; BL = T;
        
        if (j < 16) {
            T = AR + (BR ^ (CR | ~DR)) + X[ZR[j]] + K2[0];
        } else if (j < 32) {
            T = AR + ((BR & DR) | (CR & ~DR)) + X[ZR[j]] + K2[1];
        } else if (j < 48) {
            T = AR + ((BR | ~CR) ^ DR) + X[ZR[j]] + K2[2];
        } else if (j < 64) {
            T = AR + ((BR & CR) | (~BR & DR)) + X[ZR[j]] + K2[3];
        } else {
            T = AR + (BR ^ CR ^ DR) + X[ZR[j]] + K2[4];
        }
        T = ((T << SR[j]) | (T >> (32 - SR[j]))) + ER;
        AR = ER; ER = DR; DR = (CR << 10) | (CR >> 22); CR = BR; BR = T;
    }
    
    uint32_t T = h1 + CL + DR;
    h1 = h2 + DL + ER;
    h2 = h3 + EL + AR;
    h3 = h4 + AL + BR;
    h4 = h0 + BL + CR;
    h0 = T;
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        out[i]      = (h0 >> (i * 8)) & 0xFF;
        out[i + 4]  = (h1 >> (i * 8)) & 0xFF;
        out[i + 8]  = (h2 >> (i * 8)) & 0xFF;
        out[i + 12] = (h3 >> (i * 8)) & 0xFF;
        out[i + 16] = (h4 >> (i * 8)) & 0xFF;
    }
}

__device__ __forceinline__ void hash160(const uint8_t* data, int len, uint8_t out[20]) {
    uint8_t sha[32];
    sha256(data, len, sha);
    ripemd160(sha, out);
}
// Convert BigInt to hex string - optimized
__device__ void bigint_to_hex(const BigInt* bigint, char* hex_str) {
    const char hex_chars[] = "0123456789abcdef";
    int idx = 0;
    bool leading_zero = true;
    
    // Process from most significant word to least
    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        for (int j = 28; j >= 0; j -= 4) {
            uint8_t nibble = (bigint->data[i] >> j) & 0xF;
            if (nibble != 0 || !leading_zero || (i == 0 && j == 0)) {
                hex_str[idx++] = hex_chars[nibble];
                leading_zero = false;
            }
        }
    }
    
    // Handle case where number is 0
    if (idx == 0) {
        hex_str[idx++] = '0';
    }
    
    hex_str[idx] = '\0';
}
// Helper functions
__device__ __forceinline__ uint8_t get_byte(const BigInt& a, int i) {
    int word_index = 7 - (i / 4);
    int byte_index = 3 - (i % 4);
    return (a.data[word_index] >> (8 * byte_index)) & 0xFF;
}

__device__ __forceinline__ void coords_to_compressed_pubkey(const BigInt& x, const BigInt& y, uint8_t* pubkey) {
    pubkey[0] = (y.data[0] & 1) ? 0x03 : 0x02;
    #pragma unroll 8
    for (int i = 0; i < 32; i++) {
        pubkey[1 + i] = get_byte(x, i);
    }
}

__device__ __forceinline__ uint8_t hex_char_to_byte(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

__device__ void hex_string_to_bytes(const char* hex_str, uint8_t* bytes, int num_bytes) {
    #pragma unroll 8
    for (int i = 0; i < num_bytes; i++) {
        bytes[i] = (hex_char_to_byte(hex_str[i * 2]) << 4) | 
                   hex_char_to_byte(hex_str[i * 2 + 1]);
    }
}

__device__ __forceinline__ bool compare_hash160_fast(const uint8_t* hash1, const uint8_t* hash2) {
    const uint64_t* h1 = (const uint64_t*)hash1;
    const uint64_t* h2 = (const uint64_t*)hash2;
    
    return (h1[0] == h2[0]) && (h1[1] == h2[1]) && 
           (*(uint32_t*)(hash1 + 16) == *(uint32_t*)(hash2 + 16));
}

__device__ void byte_to_hex(uint8_t byte, char* out) {
    const char hex_chars[] = "0123456789abcdef";
    out[0] = hex_chars[(byte >> 4) & 0xF];
    out[1] = hex_chars[byte & 0xF];
}

__device__ void hash160_to_hex(uint8_t* hash, char* hex_str) {
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        byte_to_hex(hash[i], &hex_str[i * 2]);
    }
    hex_str[40] = '\0';
}

// Sequential search kernel with guaranteed complete coverage for 256x256 configuration
__global__ void search_private_keys_sequential(
    BigInt start_key,
    BigInt end_key,
    uint8_t* target_hash
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x; // 256 * 256 = 65,536
    
    // For 5 hex chars: 0x00000 to 0xFFFFF = 1,048,576 keys total
    const uint64_t total_keys = 0x100000; // 1,048,576
    const uint64_t keys_per_thread_base = total_keys / total_threads; // 16 keys per thread
    const uint64_t remainder = total_keys % total_threads; // 0 in this case
    
    // Calculate how many keys this thread should process
    uint64_t keys_for_this_thread;
    uint64_t start_offset;
    
    if (tid < remainder) {
        // Threads that need to process one extra key
        keys_for_this_thread = keys_per_thread_base + 1;
        start_offset = tid * keys_for_this_thread;
    } else {
        // Regular threads
        keys_for_this_thread = keys_per_thread_base;
        start_offset = remainder * (keys_per_thread_base + 1) + 
                      (tid - remainder) * keys_per_thread_base;
    }
    
    // Calculate this thread's starting key
    BigInt current_key;
    copy_bigint(&current_key, &start_key);
    
    // Add the start offset to get this thread's first key
    BigInt offset;
    init_bigint(&offset, start_offset);
    ptx_u256Add(&current_key, &current_key, &offset);
    
    // Pre-allocate working variables for performance
    ECPointJac result_jac;
    ECPoint public_key;
    uint8_t pubkey[33];
    uint8_t hash160_out[20];
    char hash160_str[41];
    char priv_key_hex[65];
    
    // Increment value (1) for moving to next key
    BigInt one;
    init_bigint(&one, 1);
    
    
    // Process exactly keys_for_this_thread keys
    for (uint64_t i = 0; i < keys_for_this_thread; i++) {
        // Early exit if another thread found the target
        if (g_found != 0) {
            return;
        }
        
        // Safety check: don't go beyond end_key
        if (compare_bigint(&current_key, &end_key) > 0) {
            // This shouldn't happen with correct range, but safety first
            if (tid == 0) {
                printf("Warning: Thread %d exceeded range at iteration %llu\n", tid, i);
            }
            return;
        }
        
        // Generate public key from private key
        scalar_multiply_jac_device(&result_jac, &const_G_jacobian, &current_key);
        jacobian_to_affine(&public_key, &result_jac);
        
        // Convert to compressed public key format
        coords_to_compressed_pubkey(public_key.x, public_key.y, pubkey);
        
        // Calculate RIPEMD160(SHA256(pubkey))
        hash160(pubkey, 33, hash160_out);
        
        
        // Check if we found the target
        if (compare_hash160_fast(hash160_out, target_hash)) {
            // Use atomic CAS to ensure only one thread reports the finding
            if (atomicCAS((int*)&g_found, 0, 1) == 0) {
                // This thread won the race to report
                bigint_to_hex(&current_key, priv_key_hex);
                hash160_to_hex(hash160_out, hash160_str);
                
                // Copy to global memory for host to retrieve
                memcpy(g_found_hex, priv_key_hex, 65);
                memcpy(g_found_hash160, hash160_str, 41);
                
                printf("\n*** FOUND BY THREAD %d! ***\n", tid);
                printf("Private Key: %s\n", priv_key_hex);
                printf("Hash160: %s\n", hash160_str);
                printf("Found at iteration %llu of %llu\n", i, keys_for_this_thread);
            }
            return;
        }
        
        // Move to the next key
        ptx_u256Add(&current_key, &current_key, &one);
    }
    
}

// Optional: Verification kernel to ensure no keys are skipped
__global__ void verify_coverage(
    BigInt start_key,
    BigInt end_key,
    uint8_t* coverage_map
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    const uint64_t total_keys = 0x100000;
    const uint64_t keys_per_thread_base = total_keys / total_threads;
    const uint64_t remainder = total_keys % total_threads;
    
    uint64_t keys_for_this_thread;
    uint64_t start_offset;
    
    if (tid < remainder) {
        keys_for_this_thread = keys_per_thread_base + 1;
        start_offset = tid * keys_for_this_thread;
    } else {
        keys_for_this_thread = keys_per_thread_base;
        start_offset = remainder * (keys_per_thread_base + 1) + 
                      (tid - remainder) * keys_per_thread_base;
    }
    
    // Mark all keys this thread would process
    for (uint64_t i = 0; i < keys_for_this_thread; i++) {
        uint64_t key_index = start_offset + i;
        if (key_index < total_keys) {
            coverage_map[key_index] = 1;
        }
    }
}

// Host function to verify coverage (call this in debug mode)
void verify_full_coverage(BigInt start_key, BigInt end_key, int blocks, int threads) {
    const size_t total_keys = 0x100000;
    uint8_t* d_coverage;
    uint8_t* h_coverage = new uint8_t[total_keys];
    
    // Allocate and initialize coverage map
    cudaMalloc(&d_coverage, total_keys * sizeof(uint8_t));
    cudaMemset(d_coverage, 0, total_keys * sizeof(uint8_t));
    
    // Run verification kernel
    verify_coverage<<<blocks, threads>>>(start_key, end_key, d_coverage);
    cudaDeviceSynchronize();
    
    // Copy back and check
    cudaMemcpy(h_coverage, d_coverage, total_keys * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    
    int missing_count = 0;
    for (size_t i = 0; i < total_keys; i++) {
        if (h_coverage[i] == 0) {
            if (missing_count < 10) { // Only print first 10 missing keys
                printf("Missing key at index: 0x%05lx\n", i);
            }
            missing_count++;
        }
    }
    
    if (missing_count == 0) {
        printf("✓ Coverage verification passed! All %zu keys will be checked.\n", total_keys);
    } else {
        printf("✗ Coverage verification failed! %d keys would be skipped.\n", missing_count);
    }
    
    // Cleanup
    cudaFree(d_coverage);
    delete[] h_coverage;
}

void init_gpu_constants() {
    // 1) 定义 p_host
    const BigInt p_host = {
        { 0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
          0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF }
    };
    // 2) 定义 G_jacobian_host
    const ECPointJac G_jacobian_host = {
        {{ 0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
                0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E }},
        {{ 0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
                0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77 }},
        {{ 1, 0, 0, 0, 0, 0, 0, 0 }}
    };
    // 3) 定义 n_host
    const BigInt n_host = {
        { 0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
          0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF }
    };

    // 然后再复制到 __constant__ 内存
    CHECK_CUDA(cudaMemcpyToSymbol(const_p, &p_host, sizeof(BigInt)));
    CHECK_CUDA(cudaMemcpyToSymbol(const_G_jacobian, &G_jacobian_host, sizeof(ECPointJac)));
    CHECK_CUDA(cudaMemcpyToSymbol(const_n, &n_host, sizeof(BigInt)));
}

void hex_to_bigint(BigInt* num, const char* hex) {
    memset(num, 0, sizeof(BigInt));
    
    int len = strlen(hex);
    int start = 0;
    
    if (len >= 2 && hex[0] == '0' && (hex[1] == 'x' || hex[1] == 'X')) {
        start = 2;
    }
    
    int word_idx = 0;
    int bit_pos = 0;
    
    for (int i = len - 1; i >= start && word_idx < BIGINT_WORDS; i--) {
        uint32_t digit_val = 0;
        char c = hex[i];
        
        if (c >= '0' && c <= '9') digit_val = c - '0';
        else if (c >= 'a' && c <= 'f') digit_val = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') digit_val = c - 'A' + 10;
        else continue;
        
        num->data[word_idx] |= (digit_val << bit_pos);
        bit_pos += 4;
        
        if (bit_pos >= 32) {
            bit_pos = 0;
            word_idx++;
        }
    }
}

void print_bigint_hex(const BigInt* num) {
    printf("0x");
    bool leading_zeros = true;
    
    for (int i = BIGINT_WORDS - 1; i >= 0; i--) {
        if (num->data[i] != 0 || !leading_zeros || i == 0) {
            if (leading_zeros) {
                printf("%x", num->data[i]);
                leading_zeros = false;
            } else {
                printf("%08x", num->data[i]);
            }
        }
    }
}

bool parse_hex_string(const char* hex, uint8_t* output, int expected_len) {
    int len = strlen(hex);
    if (len != expected_len * 2) {
        return false;
    }
    
    for (int i = 0; i < expected_len; i++) {
        char byte_str[3] = {hex[i*2], hex[i*2+1], '\0'};
        char* endptr;
        long val = strtol(byte_str, &endptr, 16);
        if (*endptr != '\0' || val < 0 || val > 255) {
            return false;
        }
        output[i] = (uint8_t)val;
    }
    return true;
}

void print_usage(const char* program_name) {
    printf("Usage: %s <start_range> <end_range> <target_hash160>\n", program_name);
    printf("\nExample:\n");
    printf("  %s 100000 1fffff 29a78213caa9eea824acf08022ab9dfc83414f56\n", program_name);
    printf("\nParameters:\n");
    printf("  start_range: Starting private key in hex (without 0x prefix)\n");
    printf("  end_range:   Ending private key in hex (without 0x prefix)\n");
    printf("  target:      Target RIPEMD160 hash (40 hex characters)\n");
}

void generate_random_prefix(char* prefix, int length) {
    for (int i = 0; i < length; i++) {
        // Get only 4 bits per digit: combine multiple rand() calls if needed
        int hex_digit = rand() & 0xF;  // only use the low 4 bits
        prefix[i] = "0123456789abcdef"[hex_digit];
    }
    prefix[length] = '\0';
}

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }
    
	
	int device_id = (argc >= 6) ? std::stoi(argv[5]) : 0;

	// Check if device exists
	int device_count = 0;
	
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return 1;
    }
	if (device_id < 0 || device_id >= device_count) {
		std::cerr << "Invalid device ID: " << device_id
				  << ". Available devices: 0 to " << (device_count - 1) << std::endl;
		return 1;
	}
	// Set device
	cudaSetDevice(device_id);
	
	std::cout << "Using CUDA device " << device_id << std::endl;
	
	
    // Seed the random number generator
    srand((unsigned)time(NULL));
	init_gpu_constants();
	precompute_G_kernel<<<1, 1>>>();
	cudaDeviceSynchronize();
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
    printf("Using GPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("\n");
    
    // Parse command line arguments
    BigInt start_key, end_key;
	int chars = (argc >= 1) ? std::stoi(argv[1]) : 13;
    int blocks = (argc >= 4) ? std::stoi(argv[3]) : 32;
    int threads = (argc >= 5) ? std::stoi(argv[4]) : 32;
	int mode = (argc >= 7) ? std::stoi(argv[6]) : 0;
    // Parse target hash160
    uint8_t target_hash[20];
    if (!parse_hex_string(argv[2], target_hash, 20)) {
        fprintf(stderr, "Error: Invalid target format: %s\n", argv[2]);
        fprintf(stderr, "Expected 40-character hex string (RIPEMD160 hash)\n");
        return 1;
    }
    
    printf("Search range:\n");
	printf("  Chars Amount: %s\n\n", argv[1]);
    printf("  Target hash160: %s\n\n", argv[2]);
    
    // Allocate device memory for target
    uint8_t* d_target_hash;
    CHECK_CUDA(cudaMalloc(&d_target_hash, 20));
    CHECK_CUDA(cudaMemcpy(d_target_hash, target_hash, 20, cudaMemcpyHostToDevice));
    
    // Start timer
    cudaEvent_t start_event, stop_event;
    CHECK_CUDA(cudaEventCreate(&start_event));
    CHECK_CUDA(cudaEventCreate(&stop_event));
    CHECK_CUDA(cudaEventRecord(start_event));
    
    printf("Starting search with %d blocks and %d threads per block\n", blocks, threads);
    printf("Total parallel threads: %d\n\n", blocks * threads);
    
	if(mode == 0)
	{
		
		printf("Mode: Random-Sequential with 256x256 configuration\n");
		printf("Each thread will process exactly 16 keys\n");
		
		//verify_full_coverage(start_key, end_key, blocks, threads);
		
		printf("Mode: Random-Sequential\n");
		while(true)
		{
			char prefix[16]; // 15 + null terminator
			prefix[0] = '1'; // Always start with '1'
			generate_random_prefix(prefix + 1, chars); // Generate rest 13 chars

			// Now generate full range
			char min_range[22];
			char max_range[22];

			snprintf(min_range, sizeof(min_range), "%s%05x", prefix, 0x00000);
			snprintf(max_range, sizeof(max_range), "%s%05x", prefix, 0xFFFFF);
			hex_to_bigint(&start_key, min_range);
			hex_to_bigint(&end_key, max_range);
			printf("Range: %s - %s\n", min_range, max_range);
			// Launch kernel
			search_private_keys_sequential<<<blocks, threads>>>(
				start_key, end_key, d_target_hash
			);
					
			CHECK_CUDA(cudaDeviceSynchronize());
			
			// Stop timer
			CHECK_CUDA(cudaEventRecord(stop_event));
			CHECK_CUDA(cudaEventSynchronize(stop_event));
			
			float milliseconds = 0;
			CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
			
			// Check results
			int h_found_flag;
			CHECK_CUDA(cudaMemcpyFromSymbol(&h_found_flag, g_found, sizeof(int)));
			
			
			
			if (h_found_flag) {
				char found_hex[65];
				char found_hash160[41];
				
				CHECK_CUDA(cudaMemcpyFromSymbol(found_hex, g_found_hex, 65));
				CHECK_CUDA(cudaMemcpyFromSymbol(found_hash160, g_found_hash160, 41));
				
				printf("\n*** MATCH FOUND! ***\n");
				printf("Private Key: %s\n", found_hex);
				printf("Hash160: %s\n", found_hash160);
				
				// Save to file
				FILE* fp = fopen("found_keys.txt", "a");
				if (fp) {
					time_t now = time(NULL);
					fprintf(fp, "[%s] Found: %s -> %s\n", ctime(&now), found_hex, found_hash160);
					fclose(fp);
					printf("Result saved to found_keys.txt\n");
				}
				// Calculate performance
				// Estimate total keys checked (this is approximate)
				BigInt range;
				uint64_t borrow = 0;
				for (int i = 0; i < 8; i++) {
					uint64_t diff = (uint64_t)end_key.data[i] - start_key.data[i] - borrow;
					range.data[i] = (uint32_t)diff;
					borrow = (diff >> 32) & 1;
				}
				
				// Simplified estimation - just use lower 64 bits for performance calc
				uint64_t total_keys = ((uint64_t)range.data[1] << 32) | range.data[0];
				if (total_keys == 0) total_keys = 1; // Avoid division by zero
				
				double keys_per_second = (double)total_keys / (milliseconds / 1000.0);
				printf("\nEstimated performance: %.2f million keys/second\n", keys_per_second / 1000000.0);
				
				// Cleanup
				CHECK_CUDA(cudaFree(d_target_hash));
				CHECK_CUDA(cudaEventDestroy(start_event));
				CHECK_CUDA(cudaEventDestroy(stop_event));
				return 0 ;
				
			}
			
		} 
	}
    
    return 0;
}