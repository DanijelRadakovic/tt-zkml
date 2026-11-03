#!/bin/bash
: "${TT_METAL_HOME:?TT_METAL_HOME must be set}"
ln -sfn "$TT_METAL_HOME/build" .tt-metal-build