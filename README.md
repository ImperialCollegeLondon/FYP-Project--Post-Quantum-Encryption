# FYP Project: Post Quantum Encryption
 Code for Final Year Master's Project at Imperial

Contents:

Report Code -> Shor's Algorithm Mock Code implemented as per: https://arxiv.org/pdf/1804.00200.pdf



## User Guide

NTRU and helper functions:       Code -> ntru.py

Reed-Muller functions:               Code -> reed_muller_coding.py

The associated NTRU functions and the functionality has been commented for all functions. This explains the functionality and the inputs and outputs. An example of a comment can be seen below. 



​    """

​    Name:        bytes_to_bits



​    Description: Converts byte array to bit array



​    Arguments:   - byte_array: Array of bytes, where elements in array in [0,255]  



​    Returns:     - bit_array: Array of bits, where elements in array are 0 or 1

​    """

The NTRU python file contains all functions required to encrypt and decrypt a message, as well as key generation methods. Example of potential implementations has also been listed, seen in the functions 'ntru_end_to_end' and 'ntru_end_to_end_reed_muller' which combines Reed-Muller encoding. Some experimental functions which include Base 3 encoding have also been listed in the python file should a user desire to implement these. Similar patterns may be followed if the NTRU function is to be implemented for testing or benchmarking in the app.

The Reed Muller python file contains 3 functions used to encode and decode message. All valid (r,m) combinations are supported as part of this library. It should be noted that the matrix generation pre-processing is necessary before any encoding and decoding takes place. This is a static operation that only runs once at the beginning, in order to save time later on in the code (at the slight expense of storage complexity). 



### DISCLAIMER: This library is only used for test purposes, and I offer no security guarantees. Please only use this for learning, testing, benchmarking purposes and not for commercial deployment of a product. 



