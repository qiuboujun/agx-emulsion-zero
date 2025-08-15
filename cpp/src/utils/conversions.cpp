#include "conversions.hpp"
#include "NumCpp.hpp"
#include "io.hpp"
#include "config.hpp"        // For load_densitometer_data
#include <cmath>         // For std::pow, std::isnan
#include <utility>       // For std::pair
#include <string>
#include <fstream>
#include <sstream>

namespace agx {
namespace utils {
/**
 * @brief Convert density to light transmittance.
 *
 * This function calculates the transmitted light intensity based on the given density and initial light intensity.
 * It uses the formula **transmittance = 10^(-density)** and then multiplies by the light intensity.
 * Any NaN values in the result are replaced with 0.
 *
 * @param density A float or nc::NdArray<float> representing the density value(s) that affect light transmittance.
 * @param light A float or nc::NdArray<float> for the initial light intensity value(s), same shape or broadcastable to `density`.
 * @return nc::NdArray<float> The light intensity after passing through the medium with the given density (same shape as input).
 */
nc::NdArray<float> density_to_light(const nc::NdArray<float>& density, const nc::NdArray<float>& light) {
    // Compute transmittance = 10^(-density)
    nc::NdArray<float> transmitted = density.copy();
    for (auto it = transmitted.begin(); it != transmitted.end(); ++it) {
        *it = std::pow(10.0f, -(*it));
    }

    // Determine broadcasting mode
    const auto rows = density.shape().rows;
    const auto cols = density.shape().cols;
    const auto lrows = light.shape().rows;
    const auto lcols = light.shape().cols;
    const auto lsize = light.size();

    if (lsize == 1) {
        // Scalar light
        const float s = light[0];
        for (auto& v : transmitted) v *= s;
    } else if (lrows == rows && lcols == cols) {
        // Exact shape match
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                transmitted(i, j) *= light(i, j);
            }
        }
    } else if (static_cast<size_t>(cols) == lsize || (lrows == 1 && lcols == cols) || (lcols == 1 && lrows == cols)) {
        // Vector length equals number of columns: broadcast across columns
        auto lflat = light.flatten();
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                transmitted(i, j) *= lflat[j];
            }
        }
    } else if (static_cast<size_t>(rows) == lsize || (lrows == rows && lcols == 1) || (lcols == rows && lrows == 1)) {
        // Vector length equals number of rows: broadcast across rows
        auto lflat = light.flatten();
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                transmitted(i, j) *= lflat[i];
            }
        }
    } else if (lsize > 1 && (cols % lsize == 0)) {
        // Block broadcast: light length K divides number of columns (W*K)
        auto lflat = light.flatten();
        const size_t blockSize = lsize;
        const size_t blocks = cols / blockSize;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t b = 0; b < blocks; ++b) {
                for (size_t k = 0; k < blockSize; ++k) {
                    transmitted(i, b*blockSize + k) *= lflat[k];
                }
            }
        }
    } else if (static_cast<size_t>(rows * cols) == lsize) {
        // Element-wise multiply by flattened light
        auto lflat = light.flatten();
        size_t idx = 0;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j, ++idx) {
                transmitted(i, j) *= lflat[idx];
            }
        }
    } else {
        // Fallback: treat as scalar using first element
        const float s = light.flatten()[0];
        for (auto& v : transmitted) v *= s;
    }

    // Replace NaN values with 0
    for (auto it = transmitted.begin(); it != transmitted.end(); ++it) {
        if (std::isnan(*it)) {
            *it = 0.0f;
        }
    }
    return transmitted;
}

// GPU-accelerated version with CPU fallback. Returns true if GPU path executed, else false.
// GPU entry points are implemented in conversions.cu
bool density_to_light_gpu(const nc::NdArray<float>&, const nc::NdArray<float>&, nc::NdArray<float>&);
bool density_to_light_cuda(const nc::NdArray<float>&, const nc::NdArray<float>&, nc::NdArray<float>&);

// Overload for single float inputs (convenience)
float density_to_light(float density, float light) {
    float transmitted = std::pow(10.0f, -density) * light;
    return std::isnan(transmitted) ? 0.0f : transmitted;
}

bool dot_blocks_K3_gpu(const nc::NdArray<float>& A,
                       const nc::NdArray<float>& B,
                       int W,
                       nc::NdArray<float>& out) {
#ifndef __CUDACC__
    return false;
#else
    const int H = static_cast<int>(A.shape().rows);
    const int K = static_cast<int>(B.shape().rows);
    out = nc::NdArray<float>(H, W*3);

    auto a = A.flatten();
    auto b = B.flatten();
    auto o = out.flatten();
    float *d_a=nullptr, *d_b=nullptr, *d_o=nullptr;
    cudaMalloc(&d_a, a.size()*sizeof(float));
    cudaMalloc(&d_b, b.size()*sizeof(float));
    cudaMalloc(&d_o, o.size()*sizeof(float));
    cudaMemcpy(d_a, a.data(), a.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), b.size()*sizeof(float), cudaMemcpyHostToDevice);

    struct Params { int H,W,K; int strideA; } p{H,W,K,(int)A.shape().cols};
    int N = H*W*3;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    __global__ void k_dot(const float* A, const float* B, float* O, Params p) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= p.H * p.W * 3) return;
        int tmp = idx;
        int c = tmp % 3; tmp /= 3;
        int w = tmp % p.W; tmp /= p.W;
        int i = tmp;
        double acc = 0.0;
        int baseA = i * p.strideA + w * p.K;
        for (int k=0;k<p.K;++k) {
            acc += (double)A[baseA + k] * (double)B[k*3 + c];
        }
        O[i*(p.W*3) + w*3 + c] = (float)acc;
    }
    k_dot<<<blocks, threads>>>(d_a, d_b, d_o, p);
    cudaDeviceSynchronize();
    cudaMemcpy(o.data(), d_o, o.size()*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_o);
    return true;
#endif
}

/**
 * @brief Compute densitometer correction factors for density measurements.
 *
 * This function computes a correction factor for each channel (assuming 3 color channels) given the dye densities and a densitometer type.
 * It uses the loaded densitometer spectral responsivities to weight the dye densities and returns 1 / (responsivity ⋅ dye_density) for each channel.
 *
 * @param dye_density An nc::NdArray<float> of shape (M,4) or similar, containing spectral dye density data. Only the first 3 columns (e.g. C, M, Y dye densities) are used.
 * @param type A string specifying the densitometer type (e.g., "status_A"). Default is "status_A".
 * @return nc::NdArray<float> A 1x3 array containing the densitometer correction factors for the three channels.
 */
nc::NdArray<float> compute_densitometer_correction(const nc::NdArray<float>& dye_density, const std::string& type) {
    // Load densitometer spectral responsivities (shape: [N,3] for R,G,B channels)
    nc::NdArray<float> responsivities = agx::utils::load_densitometer_data(type);
    // Use only the first 3 columns of dye_density (assume shape [N, >=4])
    nc::NdArray<float> dye = dye_density(nc::Slice(), nc::Slice(0, 3)).copy();
    // Replace NaNs in dye_density with 0
    for (auto it = dye.begin(); it != dye.end(); ++it) {
        if (std::isnan(*it)) {
            *it = 0.0f;
        }
    }
    // Element-wise product of responsivities and dye_density (shape [N,3])
    nc::NdArray<float> product = responsivities * dye;
    // Sum over the spectral axis (summing each column, result is 1x3)
    nc::NdArray<float> sums = nc::sum(product, nc::Axis::ROW);
    // Compute correction = 1 / sums for each channel
    nc::NdArray<float> correction(1, sums.shape().cols);
    for (size_t j = 0; j < sums.shape().cols; ++j) {
        correction(0, j) = 1.0f / sums(0, j);
    }
    return correction;
}

/**
 * @brief Computes the ACES (Academy Color Encoding System) conversion matrix for the given sensor sensitivity and illuminant.
 *
 * This function calculates the 3x3 matrix that converts from ACES2065-1 color space to the camera's raw RGB space (i.e., the ACES Input Device Transform matrix inverse).
 * It takes into account the spectral sensitivity of the camera (sensor), the illuminant spectral distribution, the CIE 1931 2° standard observer, 
 * and performs chromatic adaptation to ACES white point (D60).
 *
 * @param sensitivity nc::NdArray<float> of shape [N,3] representing the camera RGB spectral sensitivity curves.
 * @param illuminant nc::NdArray<float> of shape [N] (or [N,1]) representing the illuminant spectral power distribution (aligned to the same wavelengths as sensitivity).
 * @return nc::NdArray<float> A 3x3 matrix (nc::NdArray) that converts from ACES2065-1 RGB to raw camera RGB.
 */
nc::NdArray<float> compute_aces_conversion_matrix(const nc::NdArray<float>& sensitivity, const nc::NdArray<float>& illuminant) {
    // Dimensions check
    size_t N = sensitivity.shape().rows;
    if (sensitivity.shape().cols != 3 || illuminant.flatten().size() != N) {
        throw std::invalid_argument("Sensitivity must be N×3 and illuminant length must match N.");
    }
    nc::NdArray<float> illum = illuminant.flatten();
    
    // Use the new matrix_idt function from colour.hpp
    auto [M, RGB_w] = colour::matrix_idt(sensitivity, illum);
    
    // Invert the IDT matrix to get ACES to camera (raw) matrix
    // This is the same as np.linalg.inv(M) in Python
    float det = M(0,0)*M(1,1)*M(2,2) + M(0,1)*M(1,2)*M(2,0) + M(0,2)*M(1,0)*M(2,1)
              - M(0,2)*M(1,1)*M(2,0) - M(0,1)*M(1,0)*M(2,2) - M(0,0)*M(1,2)*M(2,1);
    
    if (std::fabs(det) < 1e-12) {
        throw std::runtime_error("Singular matrix in ACES conversion computation");
    }
    
    // Compute cofactors for inverse
    float C00 =  M(1,1)*M(2,2) - M(1,2)*M(2,1);
    float C01 = -(M(1,0)*M(2,2) - M(1,2)*M(2,0));
    float C02 =  M(1,0)*M(2,1) - M(1,1)*M(2,0);
    float C10 = -(M(0,1)*M(2,2) - M(0,2)*M(2,1));
    float C11 =  M(0,0)*M(2,2) - M(0,2)*M(2,0);
    float C12 = -(M(0,0)*M(1,2) - M(0,2)*M(1,0));
    float C20 =  M(0,1)*M(2,0) - M(0,0)*M(2,1);
    float C21 = -(M(0,1)*M(1,2) - M(0,2)*M(1,1));
    float C22 =  M(0,0)*M(1,1) - M(0,1)*M(1,0);
    
    // Adjugate transpose for inverse
    nc::NdArray<float> M_inv(3, 3);
    M_inv(0, 0) = C00 / det;
    M_inv(0, 1) = C10 / det;
    M_inv(0, 2) = C20 / det;
    M_inv(1, 0) = C01 / det;
    M_inv(1, 1) = C11 / det;
    M_inv(1, 2) = C21 / det;
    M_inv(2, 0) = C02 / det;
    M_inv(2, 1) = C12 / det;
    M_inv(2, 2) = C22 / det;
    
    return M_inv;
}

/**
 * @brief Converts RGB values to raw camera RGB values using the ACES Input Device Transform (IDT).
 *
 * This function converts an input RGB image or value from a given color space into the camera's raw RGB space, using the ACES IDT procedure.
 * It first transforms the input RGB to ACES2065-1 color space (linear AP0, D60 white), then applies the ACES-to-raw conversion matrix (from compute_aces_conversion_matrix).
 * Finally, it normalizes the output such that a mid-gray (18% reflectance) in the input becomes [1,1,1] in the raw output.
 *
 * @param RGB An nc::NdArray<float> containing the input RGB values. This can be a single RGB triplet (shape 1x3) or an array of shape [M,3] (or [H*W,3] for an image).
 * @param illuminant An nc::NdArray<float> for the illuminant spectral distribution (to compute the IDT matrix if needed).
 * @param sensitivity An nc::NdArray<float> for the camera spectral sensitivity (dimensions matching illuminant; used for IDT matrix computation).
 * @param midgray_rgb (Optional) An nc::NdArray<float> for the mid-gray RGB in the input color space. Default is [0.184, 0.184, 0.184] (18% gray) for each channel.
 * @param color_space (Optional) The color space of the input RGB values (e.g., "sRGB"). Default is "sRGB".
 * @param apply_cctf_decoding (Optional) Whether to apply the decoding of the input color component transfer function (gamma). Default is true (needed for sRGB).
 * @param aces_conversion_matrix (Optional) A precomputed 3x3 ACES-to-raw conversion matrix (from compute_aces_conversion_matrix). If not provided, it will be computed.
 * @return std::pair<nc::NdArray<float>, nc::NdArray<float>> A pair containing:
 *         - first: the raw camera RGB values (same shape as input RGB).
 *         - second: the raw mid-gray value (as a 1x3 array, typically [1,1,1]).
 */
std::pair<nc::NdArray<float>, nc::NdArray<float>> 
rgb_to_raw_aces_idt(const nc::NdArray<float>& RGB,
                    const nc::NdArray<float>& illuminant,
                    const nc::NdArray<float>& sensitivity,
                    nc::NdArray<float> midgray_rgb,
                    const std::string& color_space,
                    bool apply_cctf_decoding,
                    nc::NdArray<float> aces_conversion_matrix) {
    // Determine mid-gray values (default to 0.184 for each channel if not provided)
    float midgray_val = 0.184f;
    if (midgray_rgb.size() > 0) {
        // Use the first element of each channel if provided
        nc::NdArray<float> mg = midgray_rgb.flatten();
        if (mg.size() >= 3) {
            // Assume neutral grey, take first channel as representative (or all three if they differ)
            midgray_val = mg[0];
        }
    }
    // Convert RGB to ACES2065-1 using color science API
    // This handles CCTF decoding, color space conversion, and chromatic adaptation automatically
    nc::NdArray<float> aces = colour::RGB_to_RGB(RGB, color_space, "ACES2065-1",
                                                apply_cctf_decoding,  // Apply CCTF decoding as requested
                                                false);               // No CCTF encoding (ACES is linear)
    // Compute the ACES-to-raw conversion matrix if not provided
    nc::NdArray<float> aces_to_raw;
    if (aces_conversion_matrix.size() == 0) {
        aces_to_raw = compute_aces_conversion_matrix(sensitivity, illuminant);
    } else {
        aces_to_raw = aces_conversion_matrix;
    }
    // Multiply ACES values by aces_to_raw matrix to get raw values
    // We will perform: raw = aces @ (aces_to_raw)^T  (broadcasting each row of aces)
    nc::NdArray<float> raw = nc::dot(aces, aces_to_raw.transpose());
    // Divide by mid-gray (normalize such that input mid-gray maps to [1,1,1])
    if (raw.shape().cols == 3) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t i = 0; i < raw.shape().rows; ++i) {
                raw(i, j) /= midgray_val;
            }
        }
    } else {
        // If raw is a single 1x3 vector
        for (uint32_t j = 0; j < raw.size(); ++j) {
            raw[j] /= midgray_val;
        }
    }
    // Prepare raw_midgray array = [1,1,1]
    nc::NdArray<float> raw_midgray(1, 3);
    raw_midgray.fill(1.0f);
    return { raw, raw_midgray };
}

} // namespace utils
} // namespace agx

namespace agx { namespace utils {

nc::NdArray<float> add_glare(const nc::NdArray<float>& xyz,
                             const nc::NdArray<float>& illuminant_xyz,
                             float percent) {
    // Deterministic: add percent * illuminant_xyz uniformly to xyz
    if (percent <= 0.0f) return xyz.copy();
    const int H = static_cast<int>(xyz.shape().rows);
    const int W3 = static_cast<int>(xyz.shape().cols);
    const int W = W3 / 3;
    auto out = xyz.copy();
    float X = illuminant_xyz(0,0) * percent;
    float Y = illuminant_xyz(0,1) * percent;
    float Z = illuminant_xyz(0,2) * percent;
    for (int i=0;i<H;++i){
        for (int w=0; w<W; ++w){
            out(i, w*3 + 0) += X;
            out(i, w*3 + 1) += Y;
            out(i, w*3 + 2) += Z;
        }
    }
    return out;
}

// Simple separable Gaussian blur for a single-channel image (H x W)
static void gaussian_blur_2d(const std::vector<float>& in, int H, int W, float sigma, std::vector<float>& out, float truncate=4.0f) {
    if (sigma <= 0.0f) { out = in; return; }
    int radius = std::max(1, (int)std::ceil(truncate * sigma));
    int ksize = 2 * radius + 1;
    std::vector<float> k(ksize);
    float s2 = 2.0f * sigma * sigma; float sum = 0.0f;
    for (int i=-radius;i<=radius;++i){ float v = std::exp(-(i*i)/s2); k[i+radius]=v; sum+=v; }
    for (float &v : k) v /= sum;
    std::vector<float> tmp(H*W, 0.0f); out.assign(H*W, 0.0f);
    auto reflect = [](int x, int n){ while (x<0 || x>=n){ if (x<0) x=-x-1; else x = 2*n - x - 1; } return x; };
    // horizontal
    for (int y=0;y<H;++y){ for (int x=0;x<W;++x){ double acc=0.0; for (int t=-radius;t<=radius;++t){ int xr=reflect(x+t,W); acc += in[y*W + xr] * k[t+radius]; } tmp[y*W + x] = (float)acc; }}
    // vertical
    for (int y=0;y<H;++y){ for (int x=0;x<W;++x){ double acc=0.0; for (int t=-radius;t<=radius;++t){ int yr=reflect(y+t,H); acc += tmp[yr*W + x] * k[t+radius]; } out[y*W + x] = (float)acc; }}
}

nc::NdArray<float> add_random_glare(const nc::NdArray<float>& xyz,
                                    const nc::NdArray<float>& illuminant_xyz,
                                    float percent,
                                    float roughness,
                                    float blur_sigma_px,
                                    int height,
                                    int width,
                                    unsigned int seed) {
    if (percent <= 0.0f) return xyz.copy();
    // Build deterministic lognormal field with mean=percent and std=roughness*percent
    std::mt19937 rng(seed);
    // Compute log-space params from mean M and std S
    double M = std::max(1e-12, (double)percent);
    double S = std::max(0.0, (double)roughness * (double)percent);
    double sigma_sq = std::log(1.0 + (S*S)/(M*M));
    double sigma = std::sqrt(sigma_sq);
    double mu = std::log(M) - 0.5 * sigma_sq;
    std::lognormal_distribution<double> dist(mu, sigma);
    std::vector<float> glare(height*width);
    for (int i=0;i<height*width;++i) glare[i] = (float)dist(rng);
    // Blur
    std::vector<float> glare_blur; gaussian_blur_2d(glare, height, width, blur_sigma_px, glare_blur, 4.0f);
    // Normalize to percent in [0..1] as Python divides by 100 in model path; here percent assumed in [0..1]
    // Combine with illuminant
    auto out = xyz.copy();
    float X = illuminant_xyz(0,0), Y = illuminant_xyz(0,1), Z = illuminant_xyz(0,2);
    for (int i=0;i<height;++i){ for (int w=0; w<width; ++w){
        float g = glare_blur[i*width + w];
        out(i, w*3 + 0) += g * X;
        out(i, w*3 + 1) += g * Y;
        out(i, w*3 + 2) += g * Z;
    }}
    return out;
}

} } // namespace agx::utils