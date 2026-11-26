# What builds and what is correct

## Input format of the quantizer against the output byte encoding

Pass 2 of the 8-bit SFPU quantizer (see README.md), 48 tiles, every output byte compared with the host answer.

| input CB | twos_lib (UInt8 packer) | sm_int8pack (Int8 packer) | sm_uint8pack (UInt8 packer) |
|---|---|---|---|
| fp32 | builds, all bytes correct | builds, all bytes correct | builds, all bytes correct |
| bf16 | builds, all bytes correct | **does not build** (JIT: exponent format) | builds, all bytes correct |
| fp16 | builds, **382 wrong bytes** (inputs m = -7 -3 1 5: x * 32 = m / 2 is a tie rounded the wrong way) | builds, **382 wrong bytes** (inputs m = -7 -3 1 5: x * 32 = m / 2 is a tie rounded the wrong way) | builds, **382 wrong bytes** (inputs m = -7 -3 1 5: x * 32 = m / 2 is a tie rounded the wrong way) |

## Which other CB formats may share a program with an Int8 CB

The sm_int8pack program with fp32 input plus one unused CB (c_5) of the given format.

| extra CB format | result |
|---|---|
| fp32 | builds, all bytes correct |
| bf16 | **does not build** (JIT: exponent format) |
| fp16 | builds, all bytes correct |
| tf32 | **does not build**: what():  TF32 unsupported atm |
| bfp8_b | **does not build** (JIT: exponent format) |
| bfp4_b | **does not build** (JIT: exponent format) |
| bfp8_a | builds, all bytes correct |
| int8 | builds, all bytes correct |
| uint8 | builds, all bytes correct |
| int16 | **does not build**: TT_THROW: No valid conversion from fp32 dest to output format = Int16  |
| uint16 | builds, all bytes correct |
| int32 | builds, all bytes correct |
| uint32 | builds, all bytes correct |
