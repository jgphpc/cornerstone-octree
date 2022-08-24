/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief Thrust allocator adaptor to prevent value initialization
 *
 * Taken from: https://github.com/NVIDIA/thrust/blob/master/examples/uninitialized_vector.cu
 */

#pragma once

template<class Vector>
extern void reallocateDevice(Vector&, size_t, double);

/*! @brief resize a device vector to given number of bytes if current size is smaller
 *
 * @param[inout] vec       a device vector like thrust::device_vector
 * @param[in]    numBytes  minimum buffer size in bytes of @a vec
 * @return                 number of elements (vec.size(), not bytes) of supplied argument vector
 */
template<class Vector>
size_t reallocateDeviceBytes(Vector& vec, size_t numBytes)
{
    constexpr size_t elementSize = sizeof(typename Vector::value_type);
    size_t originalSize          = vec.size();

    size_t currentSizeBytes = originalSize * elementSize;
    if (currentSizeBytes < numBytes) { reallocateDevice(vec, (numBytes + elementSize - 1) / elementSize, 1.01); }

    return originalSize;
}