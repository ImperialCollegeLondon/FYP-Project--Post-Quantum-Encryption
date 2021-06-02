import time
import string
import random
import numpy as np


def bytes_to_bits(byte_array):

    """
    Name:        bytes_to_bits

    Description: Converts byte array to bit array

    Arguments:   - byte_array: Array of bytes, where elements in array in [0,255]  

    Returns:     - bit_array: Array of bits, where elements in array are 0 or 1
    """
    
    if max(byte_array) > 255:
        raise ValueError("Max Byte val: ", max(byte_array), " not in valid range")

    elif min(byte_array) < 0:
        raise ValueError("Min Byte val: ", min(byte_array), " not in valid range")

    bit_array = []

    for byte in byte_array:
        bit_array += [int(bit) for bit in np.binary_repr(byte,width=8)]

    return bit_array

def parity_encode(bit_array):

    """
    Name:        parity_encode

    Description: Adds single parity bit to end of array (0 or 1) depending on sum of bits

    Arguments:   - bit_array: Array of bits, where elements in array are 0 or 1  

    Returns:     - bit_array: Bit array, with parity bit at the end.
    """

    one_count = sum(bit_array)
    parity_bit = one_count % 2
    bit_array.append(parity_bit)
    return bit_array


def parity_checker(bit_array,parity_bit):

    """
    Name:        parity_checker

    Description: Checks if the parity bit received matches with the bit array

    Arguments:   - bit_array: Array of bits, where elements in array are 0 or 1  
                 - parity_bit: Single bit, 0 or 1

    Returns:     - Boolean: True if parity bit matches, else false
    """

    if sum(bit_array) % 2 == parity_bit:
        return True

    else:
        return False


ntru_chars = [50,54,74,92]

for chars in ntru_chars:
    encode_time = []
    decode_time = []
    for i in range(1000):

        random_string = (''.join(random.SystemRandom().choice(string.ascii_letters + string.digits + string.punctuation) for _ in range(50))).encode("utf-8")
        random_string_bit_stream = bytes_to_bits(random_string)

        tic = time.perf_counter()
        parity_bit_array = parity_encode(random_string_bit_stream)
        toc = time.perf_counter()
        parity_check = parity_checker(parity_bit_array[:-1],parity_bit_array[-1])
        toc_two = time.perf_counter()


        encode_time.append(toc-tic)
        decode_time.append(toc_two-toc)
        # print(toc-tic)
        # print(toc_two-tic)

    print("Encode time average", (sum(encode_time)/len(encode_time)))
    print("Decode time average", (sum(decode_time)/len(decode_time)))