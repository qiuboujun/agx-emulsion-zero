#pragma once

#include "NumCpp.hpp"
#include <vector>

namespace agx {

/**
 * @brief Performs cubic interpolation at a single point (r, g, b) in a 3D LUT.
 * This is a host-side function for single-point lookups.
 * @param lut The 3D lookup table (shape: LxLxLxC, flattened to [L*L*L, C]).
 * @param r The red coordinate (scaledx from 0 to L-1).
 * @param g The green coordinate (scaled from 0 to L-1).
 * @param b The blue coordinate (scaled from 0 to L-1).
 * @return A std::vector<float> containing the interpolated channel values.
 */
std::vector<float> cubic_interp_lut_at_3d(const nc::NdArray<float>& lut, float r, float g, float b);

/**
 * @brief Performs cubic interpolation at a single point (x, y) in a 2D LUT.
 * This is a host-side function for single-point lookups.
 * @param lut The 2D lookup table (shape: LxLxC, flattened to [L*L, C]).
 * @param x The x coordinate (scaled from 0 to L-1).
 * @param y The y coordinate (scaled from 0 to L-1).
 * @return A std::vector<float> containing the interpolated channel values.
 */
std::vector<float> cubic_interp_lut_at_2d(const nc::NdArray<float>& lut, float x, float y);

/**
 * @brief Applies a 3D LUT to an image using cubic interpolation on the GPU.
 * @param lut The 3D lookup table (shape: LxLxLxC, flattened to [L*L*L, C]).
 * @param image The input image (shape: HxWx3, flattened to [H*W, 3]).
 * @param height The original height of the image.
 * @param width The original width of the image.
 * @return A new image with the LUT applied (shape: HxWx_lut_channels, flattened to [H*W, C]).
 */
nc::NdArray<float> apply_lut_cubic_3d(const nc::NdArray<float>& lut, const nc::NdArray<float>& image, int height, int width);

/**
 * @brief Applies a 2D LUT to an image using cubic interpolation on the GPU.
 * @param lut The 2D lookup table (shape: LxLxC, flattened to [L*L, C]).
 * @param image The input image (shape: HxWxC_in, flattened to [H*W, 2]).
 * @param height The original height of the image.
 * @param width The original width of the image.
 * @return A new image with the LUT applied (shape: HxW_lut_channels, flattened to [H*W, C]).
 */
nc::NdArray<float> apply_lut_cubic_2d(const nc::NdArray<float>& lut, const nc::NdArray<float>& image, int height, int width);

} // namespace agx
