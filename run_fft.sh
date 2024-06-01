#!/bin/bash
cd build/test/
./mluop_gtest --gtest_filter=*fft*
