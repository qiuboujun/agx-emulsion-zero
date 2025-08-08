#include "grain.hpp"
#include "fast_stats.hpp"
#include <random>
#include <algorithm>
#include <numeric>
#include <cassert>

namespace {

inline int reflect_index(int i, int n) {
    if (i < 0)     return -i - 1;
    if (i >= n)    return 2*n - i - 1;
    return i;
}

static std::vector<float> gaussian_kernel_1d(float sigma) {
    if (sigma <= 0.0f) return {1.0f};
    const int radius = std::max(1, int(std::ceil(3.0f * sigma)));
    const int size   = 2*radius + 1;
    std::vector<float> k(size);
    const float s2 = 2.0f * sigma * sigma;
    float sum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        float v = std::exp(- (i*i) / s2);
        k[i + radius] = v;
        sum += v;
    }
    for (auto &v : k) v /= sum;
    return k;
}

static void convolve_h(const agx_emulsion::Image2D& src, agx_emulsion::Image2D& tmp, const std::vector<float>& k) {
    const int W = src.width, H = src.height, C = src.channels;
    const int R = (int)k.size()/2;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {
                float acc = 0.0f;
                for (int t = -R; t <= R; ++t) {
                    int xx = reflect_index(x + t, W);
                    acc += k[t+R] * src.at(xx, y, c);
                }
                tmp.at(x, y, c) = acc;
            }
        }
    }
}

static void convolve_v(const agx_emulsion::Image2D& src, agx_emulsion::Image2D& dst, const std::vector<float>& k) {
    const int W = src.width, H = src.height, C = src.channels;
    const int R = (int)k.size()/2;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {
                float acc = 0.0f;
                for (int t = -R; t <= R; ++t) {
                    int yy = reflect_index(y + t, H);
                    acc += k[t+R] * src.at(x, yy, c);
                }
                dst.at(x, y, c) = acc;
            }
        }
    }
}

inline float clamp01e6(float v) {
    const float eps = 1e-6f;
    if (v < eps) return eps;
    if (v > 1.0f - eps) return 1.0f - eps;
    return v;
}

} // namespace

namespace agx_emulsion {
namespace Grain {

void gaussian_filter_2d(const Image2D& src, Image2D& dst, float sigma) {
    if (&src == &dst) {
        Image2D tmp(src.width, src.height, src.channels);
        gaussian_filter_2d(Image2D(src), tmp, sigma);
        dst = std::move(tmp);
        return;
    }
    if (sigma <= 0.0f) {
        dst = src;
        return;
    }
    auto k = gaussian_kernel_1d(sigma);
    Image2D tmp(src.width, src.height, src.channels);
    convolve_h(src, tmp, k);
    convolve_v(tmp, dst, k);
}

Image2D layer_particle_model(const Image2D& density,
                             float density_max,
                             float n_particles_per_pixel,
                             float grain_uniformity,
                             uint64_t seed,
                             float blur_particle) {
    assert(density.channels == 1 && "layer_particle_model expects 1-channel image.");
    const int W = density.width, H = density.height;

    const float od_particle = density_max / std::max(1.0f, n_particles_per_pixel);

    std::mt19937_64 rng(seed ? seed : 5489ULL);
    Image2D out(W, H, 1);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float d = density.at(x, y, 0);
            float p = clamp01e6(d / density_max);
            float saturation = 1.0f - p * grain_uniformity * (1.0f - 1e-6f);
            float lambda = n_particles_per_pixel / std::max(1e-6f, saturation);

            std::poisson_distribution<int> pois(lambda);
            int n = std::max(0, pois(rng));

            std::binomial_distribution<int> binom(n, p);
            int developed = binom(rng);

            float val = float(developed) * od_particle * saturation;
            out.at(x, y, 0) = val;
        }
    }

    if (blur_particle > 0.0f) {
        const float sigma = blur_particle * std::sqrt(od_particle);
        Image2D tmp(W,H,1);
        gaussian_filter_2d(out, tmp, sigma);
        return tmp;
    }
    return out;
}

void apply_micro_structure(Image2D& rgb,
                           float pixel_size_um,
                           std::pair<float,float> micro,
                           uint64_t seed) {
    // micro = (blur_um, sigma_nm), convert to pixel units
    const float blur_px   = micro.first / pixel_size_um;
    const float sigma_pix = (micro.second * 0.001f) / pixel_size_um;  // (nm -> um) / pixel_size

    if (sigma_pix <= 0.05f) return;

    // Generate a lognormal random field with mu=0, sigma=sigma_pix
    std::mt19937_64 rng(seed ? seed : 1337ULL);
    std::lognormal_distribution<float> logn(0.0f, sigma_pix);

    Image2D clump(rgb.width, rgb.height, 1);
    for (int y = 0; y < rgb.height; ++y)
        for (int x = 0; x < rgb.width; ++x)
            clump.at(x,y,0) = logn(rng);

    if (blur_px > 0.4f) {
        Image2D tmp(rgb.width, rgb.height, 1);
        gaussian_filter_2d(clump, tmp, blur_px);
        clump = std::move(tmp);
    }

    // Multiply each RGB channel by the clumping field
    for (int y = 0; y < rgb.height; ++y) {
        for (int x = 0; x < rgb.width; ++x) {
            float cval = clump.at(x,y,0);
            for (int c = 0; c < rgb.channels; ++c) {
                rgb.at(x,y,c) *= cval;
            }
        }
    }
}

Image2D apply_grain_to_density(const Image2D& density_cmy,
                               float pixel_size_um,
                               float agx_particle_area_um2,
                               std::array<float,3> agx_particle_scale,
                               std::array<float,3> density_min,
                               std::array<float,3> density_max_curves,
                               std::array<float,3> grain_uniformity,
                               float grain_blur,
                               int n_sub_layers,
                               bool fixed_seed) {
    assert(density_cmy.channels == 3);

    const int W = density_cmy.width, H = density_cmy.height;
    Image2D work = density_cmy; // copy
    // Add density_min per channel
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < 3; ++c)
                work.at(x,y,c) += density_min[c];

    // Derived params
    std::array<float,3> density_max;
    for (int c = 0; c < 3; ++c) density_max[c] = density_max_curves[c] + density_min[c];

    const float pixel_area_um2 = pixel_size_um * pixel_size_um;
    std::array<float,3> agx_particle_area_um2_rgb;
    for (int c = 0; c < 3; ++c) agx_particle_area_um2_rgb[c] = agx_particle_area_um2 * agx_particle_scale[c];
    std::array<float,3> npp;
    for (int c = 0; c < 3; ++c) npp[c] = pixel_area_um2 / agx_particle_area_um2_rgb[c];

    if (n_sub_layers > 1) {
        for (int c = 0; c < 3; ++c) npp[c] /= float(n_sub_layers);
    }

    Image2D out(W,H,3);
    for (int ch = 0; ch < 3; ++ch) {
        Image2D acc(W,H,1); // accumulator for sublayers
        for (int sl = 0; sl < n_sub_layers; ++sl) {
            // Extract channel to 1-channel density
            Image2D d1(W,H,1);
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x)
                    d1.at(x,y,0) = work.at(x,y,ch);

            uint64_t seed = fixed_seed ? 0ULL : (uint64_t)(ch + sl*10);
            Image2D g = layer_particle_model(d1, density_max[ch], npp[ch], grain_uniformity[ch], seed, 0.0f);

            // Accumulate
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x)
                    acc.at(x,y,0) += g.at(x,y,0);
        }
        // Average sublayers
        const float inv = 1.0f / std::max(1, n_sub_layers);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                out.at(x,y,ch) = acc.at(x,y,0) * inv;
    }

    // Subtract density_min
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < 3; ++c)
                out.at(x,y,c) -= density_min[c];

    // Optional blur across X,Y only
    if (grain_blur > 0.4f) {
        Image2D tmp(W,H,3);
        gaussian_filter_2d(out, tmp, grain_blur);
        out = std::move(tmp);
    }

    return out;
}

Image2D apply_grain_to_density_layers(const Image2D& density_cmy_layers,
                                      const std::array<std::array<float,3>,3>& density_max_layers,
                                      float pixel_size_um,
                                      float agx_particle_area_um2,
                                      std::array<float,3> agx_particle_scale,
                                      std::array<float,3> agx_particle_scale_layers,
                                      std::array<float,3> density_min,
                                      std::array<float,3> grain_uniformity,
                                      float grain_blur,
                                      float grain_blur_dye_clouds_um,
                                      std::pair<float,float> grain_micro_structure,
                                      bool fixed_seed,
                                      bool use_fast_stats) {
    assert(density_cmy_layers.channels == 9); // 3 sublayers x 3 channels (sl major)

    const int W = density_cmy_layers.width, H = density_cmy_layers.height;

    // Compute total max and fractions (per channel)
    std::array<float,3> density_max_total = {0,0,0};
    for (int sl = 0; sl < 3; ++sl)
        for (int ch = 0; ch < 3; ++ch)
            density_max_total[ch] += density_max_layers[sl][ch];

    std::array<std::array<float,3>,3> density_max_frac{};
    for (int sl = 0; sl < 3; ++sl)
        for (int ch = 0; ch < 3; ++ch)
            density_max_frac[sl][ch] = density_max_layers[sl][ch] / std::max(1e-6f, density_max_total[ch]);

    // density_min shares fractions
    std::array<std::array<float,3>,3> density_min_layers{};
    for (int sl = 0; sl < 3; ++sl)
        for (int ch = 0; ch < 3; ++ch)
            density_min_layers[sl][ch] = density_max_frac[sl][ch] * density_min[ch];

    // Add min to max for layer-wise absolute maxima
    std::array<std::array<float,3>,3> density_max_abs = density_max_layers;
    for (int sl = 0; sl < 3; ++sl)
        for (int ch = 0; ch < 3; ++ch)
            density_max_abs[sl][ch] += density_min_layers[sl][ch];

    // Particles per pixel per layer/channel
    const float pixel_area_um2 = pixel_size_um * pixel_size_um;
    std::array<std::array<float,3>,3> agx_area{};
    for (int sl = 0; sl < 3; ++sl)
        for (int ch = 0; ch < 3; ++ch)
            agx_area[sl][ch] = agx_particle_area_um2 * agx_particle_scale[ch] * agx_particle_scale_layers[sl];

    std::array<std::array<float,3>,3> npp{};
    for (int sl = 0; sl < 3; ++sl)
        for (int ch = 0; ch < 3; ++ch)
            npp[sl][ch] = pixel_area_um2 * density_max_frac[sl][ch] / std::max(1e-6f, agx_area[sl][ch]);

    // Add layer-wise density_min
    Image2D work = density_cmy_layers;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            for (int sl = 0; sl < 3; ++sl)
                for (int ch = 0; ch < 3; ++ch)
                    work.at(x,y,sl + 3*ch) += density_min_layers[sl][ch];

    Image2D out(W,H,3); // result RGB
    for (int ch = 0; ch < 3; ++ch) {
        Image2D acc(W,H,1);
        for (int sl = 0; sl < 3; ++sl) {
            // extract this layer+channel
            Image2D d1(W,H,1);
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x)
                    d1.at(x,y,0) = work.at(x,y,sl + 3*ch);

            uint64_t seed = fixed_seed ? 0ULL : (uint64_t)(ch + sl*10);
            Image2D g = layer_particle_model(d1,
                                             density_max_abs[sl][ch],
                                             npp[sl][ch],
                                             grain_uniformity[ch],
                                             seed,
                                             grain_blur_dye_clouds_um);
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x)
                    acc.at(x,y,0) += g.at(x,y,0);
        }
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                out.at(x,y,ch) = acc.at(x,y,0);
    }

    // Micro-structure
    apply_micro_structure(out, pixel_size_um, grain_micro_structure, /*seed*/777);

    // Use FastStats for statistics if requested
    if (use_fast_stats) {
        for (int ch = 0; ch < 3; ++ch) {
            std::vector<float> channel_data;
            channel_data.reserve(W * H);
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    channel_data.push_back(out.at(x, y, ch));
                }
            }
            auto [mean, stddev] = FastStats::mean_stddev(channel_data);
            // Could use these statistics for additional processing if needed
        }
    }

    // Final subtract density_min and optional blur
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            for (int ch = 0; ch < 3; ++ch)
                out.at(x,y,ch) -= density_min[ch];

    if (grain_blur > 0.0f) {
        Image2D tmp(W,H,3);
        gaussian_filter_2d(out, tmp, grain_blur);
        out = std::move(tmp);
    }

    return out;
}

} // namespace Grain
} // namespace agx_emulsion 