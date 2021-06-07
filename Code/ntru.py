#Main math imports
import os
# from base64 import b64encode
import numpy as np
import random
import string
import math
import time
import reed_muller_coding

#Sympy imports (used for poly space)
from sympy import ZZ, Poly, invert, GF, isprime
from sympy.abc import x
from sympy.polys.polyerrors import NotInvertible



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



def bytes_to_ternary(byte_array):

    """
    Name:        bytes_to_ternary

    Description: Converts byte array to ternary array

    Arguments:   - byte_array: Array of bytes, where elements in array in [0,255]  

    Returns:     - ternary_array: Array of trits, where elements in array are -1, 0 or 1
    """
    

    if max(byte_array) > 255:
        raise ValueError("Max Byte val: ", max(byte_array), " not in valid range")

    elif min(byte_array) < 0:
        raise ValueError("Min Byte val: ", min(byte_array), " not in valid range")

    ternary_array = []
    for byte in byte_array:
        for j in range(5,-1,-1):
            if byte >= 2 * 3**j:
                ternary_array.append(1)
                byte -= 2 * 3**j
            elif 3**j <= byte <= 2*3**j:
                ternary_array.append(0)
                byte -= 3**j
            else:
                ternary_array.append(-1)

    return ternary_array



def bits_to_bytes(bit_array,check):

    """
    Name:        bits_to_bytes

    Description: Converts bit array to a byte value

    Arguments:   - bit_array: Array of bits, where elements in array in 0 or 1
                 - check: Boolean whether to check if byte value is valid or not. 
                            As this function is reused for ECC decoding where
                            valid byte value may not matter due to different base.  

    Returns:     - byte_value: Byte value in range [0,255]
    """

    byte_value = 0
    two_power = 1

    for bit in bit_array[::-1]:
        byte_value += two_power * bit
        two_power = two_power * 2

    if check and byte_value > 255:
        raise ValueError("Max Byte val: ", max(byte_array), " not in valid range")

    return byte_value


def trits_to_bytes(bit_array):

    """
    Name:        trits_to_bytes

    Description: Converts ternary array to a byte value

    Arguments:   - bit_array: Array of bits, where elements in array are -1,0 or 1

    Returns:     - byte_value: Byte value in range [0,255]
    """

    byte_value = 0
    three_power = 1

    for bit in bit_array[::-1]:
        byte_value += three_power * (bit+1)
        three_power = three_power * 3


    if byte_value > 255:
        raise ValueError("Max Byte val: ", max(byte_array), " not in valid range")

    return byte_value


def bit_padding(bit_array,num_of_bits):

    """
    Name:        bit_padding

    Description: Adds padding to bit array, such that length of array is a multiple of 'num_of_bits'

    Arguments:   - bit_array: Array of bits, where elements in array are 0 or 1  
                 - num_of_bits: Length of array should be multiple of this. 8 for base 2, 6 for base 3

    Returns:     - bit_array: Bit array, where elements in array are 0 or 1 with padding of value 0
    """

    excess = len(bit_array) % num_of_bits

    if excess == 0:
        return bit_array

    else:
        bit_array = [0]*(num_of_bits-excess) + bit_array
        return bit_array


def string_decode(bit_array):

    """
    Name:        string_decode

    Description: Decode the string, from bytes (in UTF-8 format) to a string

    Arguments:   - bit_array: Array of bits, where elements in array are 0 or 1

    Returns:     - output_string: Text string containing the message
    """

    output_string = ""
    for i in range(0,int(len(bit_array)/8)):
        output_string += bytes([bits_to_bytes(bit_array[i*8:(i+1)*8],True)]).decode("utf-8")

    return output_string



def string_decode_ternary(bit_array):

    """
    Name:        string_decode_ternary

    Description: Decode the string, from bytes (in UTF-8 format) to a string

    Arguments:   - bit_array: Array of bits, where elements in array are -1, 0 or 1

    Returns:     - output_string: Text string containing the message
    """

    output_string = ""
    for i in range(0,int(len(bit_array)/6)):
        output_string += bytes([trits_to_bytes(bit_array[i*6:(i+1)*6])]).decode("utf-8")

    return output_string


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


def encrypted_message_encoder(encrypted_message,n):

    """
    Name:        encrypted_message_encoder

    Description: Appends 0 to coefficients array, if the leading cofficient values are 0

    Arguments:   - encrypted_message: Array of bits, where elements in array are 0 or 1  
                 - n: Length the array should be

    Returns:     - encrypted_message: Appended with leading zeros if required.
    """


    if n > len(encrypted_message):
        raise RuntimeError("Check input for n or encrypted message. Not Valid")

    else:
        return [0]*(n-len(encrypted_message)) + encrypted_message


def number_to_binary(number_array,bits):

    """
    Name:        number_to_binary

    Description: Converts a number to binary representation with a given bit range.

    Arguments:   - number_array: Array of numbers to be encoded into bits.
                 - bits: Length of bit array for each number

    Returns:     - bit_array: Array of bits, where elements in array are 0 or 1
    """

    bit_array = []

    for num in number_array:
        bit_array += [int(bit) for bit in np.binary_repr(num,width=bits)]

    return bit_array

def binary_to_number(bit_array,bits):

    """
    Name:       binary_to_number

    Description: Converts binary to number representation with a given bit range.

    Arguments:   - binary_array: Array of bits, where elements in array are 0 or 1 to be decoded into numbers.
                 - bits: Length of bit array for each number

    Returns:     - bit_array: Array of numbers
    """

    number_array = []

    for i in range(int(len(bit_array)/bits)):
        number_array.append(bits_to_bytes(bit_array[i*bits:(i+1)*bits], False))

    return number_array


class ntru():
    n = None
    p = None
    q = None

    f = None

    f_p = None
    f_q = None

    g = None

    r_space = None

    h = None

    def __init__(self,n,p,q):


        """
        Name:        init

        Description: Initialiser for NTRU class, and defines r space for given parameters

        Arguments:   - n: NTRU Parameter  
                     - p: NTRU Parameter
                     - q: NTRU Parameter

                     Where n is prime, p and q are coprime.

        Returns:     - Boolean: True if parity bit matches, else false
        """

        self.n = n
        self.p = p
        self.q = q
        self.r_space = Poly(x**n - 1, x).set_domain(ZZ)


    def g_gen(self):

        """
        Name:        g_gen

        Description: Generates random polynomial g, used to create a random public key

        Arguments:  None (values taken from class instance)

        Returns:     - g: Random polynomial of maximum degree n
        """
        #num_ones = math.floor(self.n/3)
        num_ones = int(math.sqrt(self.q))
        num_zeros = self.n - 2*num_ones

        g_factors = np.random.permutation(np.concatenate((np.zeros(num_zeros), np.ones(num_ones), -np.ones(num_ones))))
        
        return Poly(g_factors,x).set_domain(ZZ)

    def r_gen(self):

        """
        Name:        r_gen

        Description: Generates random polynomial r, used to encrypt the message

        Arguments:   None (values taken from class instance)

        Returns:     - r: Random polynomial of maximum degree n
        """


        r_factors = 0
        for i in range(0,self.n):
            r_factors += random.randint(-1,1) * x**i

        r = Poly(r_factors,x).set_domain(ZZ)

        return r


    def f_invert(self,f,mod_param):

        """
        Name:        f_invert

        Description: Calculates the inverse of f modulo mod_param. Used to form public key.

        Arguments:   - f: Polynomial of maximum degree n  
                     - mod_param: Integer for which the inverse of f modulo calculated. mod_param = p or q

        Returns:     - f_inv: Inverse of f modulo mod_param
        """

        if math.log(mod_param,2) == int(math.log(mod_param,2)): 
            f_inv = invert(f,self.r_space, domain=GF(2,symmetric = False))

            log_val = int(math.log(mod_param, 2))

            for i in range(1, log_val):
                f_inv = ((2 * f_inv - f * f_inv ** 2) % self.r_space).trunc(mod_param)
            return f_inv
        else:
            return invert(f,self.r_space,domain=GF(mod_param))


    def f_gen(self,max_attempts):

        """
        Name:        f_gen

        Description: Generates random polynomial f, used to form private/public keys.

        Arguments:   - max_attempts: Number of attempts to find f such that inverse of f mod p and q exists 


        Returns:     - f_inv: Random polynomial f satisfying above criteria
                     - RunTimeError: If inverse of f cannot be found in the given max attempts
        """

        attempts = 0

        f_valid = False

        while(attempts < max_attempts and f_valid == False):
            f_factors = 0
            for i in range(0,self.n):
                f_factors += random.randint(-1,1) * x**i


            f = Poly(f_factors,x).set_domain(ZZ)

            try:
                f_inv_p = self.f_invert(f,self.p)
                f_inv_q = self.f_invert(f,self.q)
                f_valid = True

            except:
                attempts = attempts + 1


        if f_valid == False:
            raise RuntimeError("Could not find a suitable f in attempt constraints, check inputs or try again")

        else:
            self.f_p = f_inv_p
            self.f_q = f_inv_q
            return f

    def key_gen(self):

        """
        Name:        key_gen

        Description: Key generator function, calls g_gen, f_gen and h_gen to generate the necessary keys

        Arguments:   None (values taken from class instance)

        Returns:     None (stores in class instance)
        """

        self.g = self.g_gen()

        self.f = self.f_gen(20) #can change max attempts here, set to 20 by default as this was sufficient in early ntru_instanceing

        self.h = (((self.p * self.f_q).trunc(self.q) * self.g).trunc(self.q) % self.r_space).trunc(self.q)


    def encrypt(self,m):

        """
        Name:        encrypt

        Description: Encrypts the given message

        Arguments:   - m: Message to be encrypted in bit array (0 or 1) 

        Returns:     - e: Encrypted message (as SYMPY polynomial)
        """

        r = self.r_gen()

        r_h = (r * self.h).trunc(self.q)

        e = ((r_h + m) % self.r_space).trunc(self.q)

        return e


    def decrypt(self,e):

        """
        Name:        decrypt

        Description: Decrypts the message

        Arguments:   - e: Encrypted message (as SYMPY polynomial)  

        Returns:     - c: Decrypted message in bit array (0 or 1)
        """

        a = ((self.f * e) % self.r_space).trunc(self.q)
        b = a.trunc(self.p)

        c = ((self.f_p * b) % self.r_space).trunc(self.p)

        return c



def ntru_end_to_end(message_string, n = 401, p = 3 , q = 2048, detailed_stats = False):

    """
    Name:        ntru_end_to_end

    Description: End to end tester of NTRU Encrypt/Decrypt. Takes in message string, encrypts
                 and decrypts. Used in testing of performance and verification of implementation.

    Arguments:   - message_string: Message (string) to be encrypted
                 - n: NTRU Parameter, set to 401 by default  
                 - p: NTRU Parameter, set to 3 by default
                 - q: NTRU Parameter, set to 2048 by default.
                 - detailed_stats: If True, a dictionary containing public key information is returned.
                   Set to False by default. 

                 Where n is prime, p and q are coprime.

    Returns:     - message_string: Returns the original message string.
                 - detailed_stats_dict: Optional, if detailed_stats is specified is True. 
                   Contains details about the encryption, including the keys used.
    """
    
    ntru_instance = ntru(n,p,q)
    ntru_instance.key_gen()
    full_string_length = len(message_string)

    decoded_full_string = ""

    max_chars = int(math.floor(n/8))

    if len(message_string) % max_chars == 0:
        splits = int(math.floor(full_string_length/max_chars)) #define to be 8 here because UTF-8 256 is used. 

    else:
        splits = splits = int(math.floor(full_string_length/max_chars)) + 1

    for i in range(0,splits):

        partial_msg_string = message_string[(i*max_chars) : min(len(message_string),(i+1)*max_chars)]
        
        encoded_string = partial_msg_string.encode("utf-8")

        byte_list = list(encoded_string)

        bit_list = bytes_to_bits(byte_list)

        original_m = Poly(bit_list,x).set_domain(ZZ)

        encrypted_m = ntru_instance.encrypt(original_m)

        decrypted_m = ntru_instance.decrypt(encrypted_m)

        coeffs = bit_padding(decrypted_m.all_coeffs(),8)

        decoded_string = string_decode(coeffs)

        decoded_full_string += decoded_string


    if detailed_stats:
        detailed_stats_dict = {"f": ntru_instance.f, "g": ntru_instance.g, "f_p": ntru_instance.f_p, "f_q": ntru_instance.f_q}
        return(detailed_stats_dict,decoded_full_string)
    
    else:
        return(decoded_full_string)



def ntru_end_to_end_reed_muller(message_string, n = 401, p = 3, detailed_stats = False):

    """
    Name:        ntru_end_to_end_reed_muller

    Description: End to end tester of NTRU Encrypt/Decrypt. Takes in message string, encrypts
                 and decrypts. Used in testing of performance and verification of implementation.

                 Added reed muller encoding and decoding inbetween encryption and decryption of numbers.

    Arguments:   - message_string: Message (string) to be encrypted
                 - n: NTRU Parameter, set to 401 by default  
                 - p: NTRU Parameter, set to 3 by default
                 - detailed_stats: If True, a dictionary containing public key information is returned.
                   Set to False by default. 

                   NOTE: Q MUST BE 2048 WITH THIS IMPLEMENTATION.

                 Where n is prime, p and q are coprime.

    Returns:     - message_string: Returns the original message string.
                 - detailed_stats_dict: Optional, if detailed_stats is specified is True. 
                   Contains details about the encryption, including the keys used.
    """
    
    q = 2048
    
    #Defining NTRU Parameters:
    r = 2
    m = 4
    reed_muller_gen_matrix,reed_muller_inverse_matrix,val_list = reed_muller_coding.matrix_gen(r, m)

    ntru_instance = ntru(n,p,q)
    ntru_instance.key_gen()
    full_string_length = len(message_string)

    decoded_full_string = ""

    max_chars = int(math.floor(n/8))

    if len(message_string) % max_chars == 0:
        splits = int(math.floor(full_string_length/max_chars)) #define to be 8 here because UTF-8 256 is used. 

    else:
        splits = splits = int(math.floor(full_string_length/max_chars)) + 1

    for i in range(0,splits):

        partial_msg_string = message_string[(i*max_chars) : min(len(message_string),(i+1)*max_chars)]
        
        encoded_string = partial_msg_string.encode("utf-8")

        byte_list = list(encoded_string)

        bit_list = bytes_to_bits(byte_list)

        original_m = Poly(bit_list,x).set_domain(ZZ)

        encrypted_m = ntru_instance.encrypt(original_m)
        encrypted_coeffs = encrypted_message_encoder(encrypted_m.all_coeffs(),n)

        positive_coeffs = [int(i+(q/2)-1) for i in encrypted_coeffs]

        encrypted_bits = number_to_binary(positive_coeffs,int(math.log(q,2)))

        decoded_bits = []

        encode_time = 0
        decode_time = 0

        for i in range(int(len(encrypted_bits)/11)):
            tic = time.perf_counter()
            reed_muller_bits = reed_muller_coding.encode(encrypted_bits[i*11:(i+1)*11], r, m, reed_muller_gen_matrix)
            toc = time.perf_counter()
            decoded_bits += reed_muller_coding.decode(reed_muller_bits, r, m, reed_muller_gen_matrix, reed_muller_inverse_matrix, val_list)
            toc_two = time.perf_counter()

            encode_time += (toc-tic)
            decode_time += (toc_two-toc)

        print(encode_time)
        print(decode_time)

        decoded_positive_coeffs = binary_to_number(decoded_bits,int(math.log(q,2)))

        decoded_coeffs = [int(i-(q/2)+1) for i in decoded_positive_coeffs]

        decrypted_m = ntru_instance.decrypt(Poly(decoded_coeffs,x).set_domain(ZZ))

        coeffs = bit_padding(decrypted_m.all_coeffs(),8)

        decoded_string = string_decode(coeffs)

        decoded_full_string += decoded_string


    return (encode_time,decode_time)

    # if detailed_stats:
    #     detailed_stats_dict = {"f": ntru_instance.f, "g": ntru_instance.g, "f_p": ntru_instance.f_p, "f_q": ntru_instance.f_q}
    #     return(detailed_stats_dict,decoded_full_string)
    
    # else:
    #     return(decoded_full_string)


def ntru_aes_package(aes_size=256,n=401,p=3,q=2048, detailed_stats = False):

    """
    Name:        ntru_aes_package

    Description: End to end tester of NTRU Encrypt/Decrypt, with AES symmetric key as payload.
                 Generates random AES key, encrypts and decrypts. Used in testing of performance and verification of implementation.

    Arguments:   - aes_size: Size of AES payload. 128,192 and 256 bit supported.
                 - n: NTRU Parameter, set to 401 by default  
                 - p: NTRU Parameter, set to 3 by default
                 - q: NTRU Parameter, set to 2048 by default
                 - detailed_stats: If True, a dictionary containing public key information is returned.
                   Set to False by default. 

                 Where n is prime, p and q are coprime.

    Returns:     - message_string: Returns the original message string.
                 - detailed_stats_dict: Optional, if detailed_stats is specified is True. 
                   Contains details about the encryption, including the keys used.
                 - ValueError if AES size not valid.
    """

    valid_aes_sizes = [128,192,256]
    if aes_size not in valid_aes_sizes:
        raise ValueError("AES size has to be 128, 192 or 256 bits")

    aes_key = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits + string.punctuation) for _ in range(int(aes_size/8)))

    aes_key_byte_rep = aes_key.encode("utf-8")

    print(aes_key)

    return (ntru_end_to_end(aes_key, n, p, q, detailed_stats),aes_key_byte_rep)
    
    

def key_gen(n=401,p=3,q=2048):

    """
    Name:        key_gen

    Description: Function to generate NTRU keys.

    Arguments:   - n: NTRU Parameter, set to 401 by default  
                 - p: NTRU Parameter, set to 3 by default
                 - q: NTRU Parameter, set to 2048 by default

                 Where n is prime, p and q are coprime.

    Returns:     - dictionary: Returns a dictionary containing key values: g,f,h,f_p and f_q
    """
    ntru_instance = ntru(n,p,q)
    ntru_instance.key_gen()

    return {"g":ntru_instance.g, "f": ntru_instance.f, "h": ntru_instance.h, "f_p": ntru_instance.f_p, "f_q": ntru_instance.f_q}



def encrypt(message_string,h,n=401,p=3,q=2048):

    """
    Name:        encrypt

    Description: Function to encrypt a given message.

    Arguments:   - message_string: Message to be encrypted
                 - n: NTRU Parameter, set to 401 by default  
                 - p: NTRU Parameter, set to 3 by default
                 - q: NTRU Parameter, set to 2048 by default
                 - h: Public key 

                 Where n is prime, p and q are coprime.

    Returns:     - encrypted_bits: Encrypted message as bits
    """

    ntru_instance = ntru(n,p,q)
    ntru_instance.h = h

    full_string_length = len(message_string)

    max_chars = int(math.floor(n/8))

    encrypted_polynomial_arr = []

    if len(message_string) % max_chars == 0:
        splits = int(math.floor(full_string_length/max_chars)) #define to be 8 here because UTF-8 256 is used. 

    else:
        splits = splits = int(math.floor(full_string_length/max_chars)) + 1

    for i in range(0,splits):

        partial_msg_string = message_string[(i*max_chars) : min(len(message_string),(i+1)*max_chars)]
        
        encoded_string = partial_msg_string.encode("utf-8")

        byte_list = list(encoded_string)

        bit_list = bytes_to_bits(byte_list)

        original_m = Poly(bit_list,x).set_domain(ZZ)

        encrypted_m = ntru_instance.encrypt(original_m)

        encrypted_polynomial_arr.append(encrypted_m)


    return encrypted_polynomial_arr


def decrypt(encrypted_bits,f,f_p,n=401,p=3,q=2048):

    """
    Name:        decrypt

    Description: Function to decrypt a given message.

    Arguments:   - message_string: Message to be encrypted
                 - f: NTRU Polynomial, private key
                 - f_p: NTRU Polynomial, private key
                 - n: NTRU Parameter, set to 401 by default  
                 - p: NTRU Parameter, set to 3 by default
                 - q: NTRU Parameter, set to 2048 by default
                 - h: Public key 

                 Where n is prime, p and q are coprime.

    NOTE: This function is currently experimental and only supports
    block by block decrypting at this stage. 

    Returns:     - encrypted_bits: Encrypted message as bits
    """

    ntru_instance = ntru(n,p,q)
    ntru_instance.f = f
    ntru_instance.f_p = f_p

    encrypted_m = encrypted_bits

    decrypted_m = ntru_instance.decrypt(encrypted_m)

    coeffs = bit_padding(decrypted_m.all_coeffs(),8)

    decoded_string = string_decode(coeffs)

    return decoded_string




def ntru_end_to_end_ternary(message_string, n = 401, p = 3 , q = 2048, detailed_stats = False):

    """
    Name:        ntru_end_to_end_ternary

    Description: End to end tester of NTRU Ternary Encrypt/Decrypt (base 3). Takes in message string, encrypts
                 and decrypts. Used in testing of performance and verification of implementation.

                 Unlike other NTRU functions, this encodes the character bytes into base 3 instead of base 2.and
                 This function is experimental, and not part of the original NTRU algorithm but rather for innovation in this project.

    Arguments:   - message_string: Message (string) to be encrypted
                 - n: NTRU Parameter, set to 401 by default  
                 - p: NTRU Parameter, set to 3 by default
                 - q: NTRU Parameter, set to 2048 by default
                 - detailed_stats: If True, a dictionary containing public key information is returned.
                   Set to False by default. 

                 Where n is prime, p and q are coprime.

    Returns:     - message_string: Returns the original message string.
                 - detailed_stats_dict: Optional, if detailed_stats is specified is True. 
                   Contains details about the encryption, including the keys used.
    """
    
    ntru_instance = ntru(n,p,q)
    ntru_instance.key_gen()
    full_string_length = len(message_string)

    decoded_full_string = ""

    max_chars = int(math.floor(n/6))

    if len(message_string) % max_chars == 0:
        splits = int(math.floor(full_string_length/max_chars)) #define to be 8 here because UTF-8 256 is used. 

    else:
        splits = splits = int(math.floor(full_string_length/max_chars)) + 1

    for i in range(0,splits):

        partial_msg_string = message_string[(i*max_chars) : min(len(message_string),(i+1)*max_chars)]
        
        encoded_string = partial_msg_string.encode("utf-8")

        byte_list = list(encoded_string)

        bit_list = bytes_to_ternary(byte_list)

        original_m = Poly(bit_list,x).set_domain(ZZ)

        encrypted_m = ntru_instance.encrypt(original_m)
        decrypted_m = ntru_instance.decrypt(encrypted_m)

        coeffs = bit_padding(decrypted_m.all_coeffs(),6)

        decoded_string = string_decode_ternary(coeffs)

        decoded_full_string += decoded_string


    if detailed_stats:
        detailed_stats_dict = {"f": ntru_instance.f, "g": ntru_instance.g, "f_p": ntru_instance.f_p, "f_q": ntru_instance.f_q}
        return(detailed_stats_dict,decoded_full_string)
    
    else:
        return(decoded_full_string)



def ntru_with_parity(message_string, n = 401, p = 3 , q = 2048, detailed_stats = False):

    """
    Name:        ntru_with_parity

    Description: End to end tester of NTRU Encrypt/Decrypt with added parity bit. Takes in message string, encrypts
                 and decrypts. Used in testing of performance and verification of implementation.

                 In addition to the base NTRU function, the message payload contains an extra parity bit.
                 This parity bit is then checked in the decryption process, as a verification check.

    Arguments:   - message_string: Message (string) to be encrypted
                 - n: NTRU Parameter, set to 401 by default  
                 - p: NTRU Parameter, set to 3 by default
                 - q: NTRU Parameter, set to 2048 by default
                 - detailed_stats: If True, a dictionary containing public key information is returned.
                   Set to False by default. 

                 Where n is prime, p and q are coprime.

    Returns:     - message_string: Returns the original message string.
                 - detailed_stats_dict: Optional, if detailed_stats is specified is True. 
                   Contains details about the encryption, including the keys used.
    """
    
    ntru_instance = ntru(n,p,q)
    ntru_instance.key_gen()
    full_string_length = len(message_string)

    decoded_full_string = ""

    max_chars = int(math.floor(n/8))

    if len(message_string) % max_chars == 0:
        splits = int(math.floor(full_string_length/max_chars)) #define to be 8 here because UTF-8 256 is used. 

    else:
        splits = splits = int(math.floor(full_string_length/max_chars)) + 1

    for i in range(0,splits):
        partial_msg_string = message_string[(i*max_chars) : min(len(message_string),(i+1)*max_chars)]
        
        encoded_string = partial_msg_string.encode("utf-8")

        byte_list = list(encoded_string)

        bit_list = bytes_to_bits(byte_list)

        bit_list_with_parity = parity_encode(bit_list)

        original_m = Poly(bit_list_with_parity,x).set_domain(ZZ)

        encrypted_m = ntru_instance.encrypt(original_m)
        decrypted_m = ntru_instance.decrypt(encrypted_m)

        parity_bit = decrypted_m.all_coeffs()[-1]
        coeffs = bit_padding(decrypted_m.all_coeffs()[:-1],8)

        if parity_checker(coeffs,parity_bit) == False:
            raise RuntimeError("Error in transmission, re-transmit message.")

        decoded_string = string_decode(coeffs)

        decoded_full_string += decoded_string


    if detailed_stats:
        detailed_stats_dict = {"f": ntru_instance.f, "g": ntru_instance.g, "f_p": ntru_instance.f_p, "f_q": ntru_instance.f_q}
        return(detailed_stats_dict,decoded_full_string)
    
    else:
        return(decoded_full_string)


def aes_generator(aes_size=256):

    """
    Name:        aes_generator

    Description: Generates a random string which can be used as an AES key.

    Arguments:   - aes_size: Size of AES payload. 128,192 and 256 bit supported. Set to 256 bits by default.

    Returns:     - aes_key: Returns an AES key for the size specified.
    """

    valid_aes_sizes = [128,192,256]
    if aes_size not in valid_aes_sizes:
        raise ValueError("AES size has to be 128, 192 or 256 bits")

    aes_key = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits + string.punctuation) for _ in range(int(aes_size/8)))

    return aes_key



print("Test Function Area")




# print(ntru_end_to_end_reed_muller("banana", n=167,p=3))



print("Reed Muller Encode/Decode Testing")

n_values = [409,439,593,743]

for n_val in n_values:
    encode_times = []
    decode_times = []

    for i in range(100):
        encode_time,decode_time = ntru_end_to_end_reed_muller("test",n=n_val)
        encode_times.append(encode_time)
        decode_times.append(decodE_time)

    print("Averages for: ", n_val)
    print("Encode Average", sum(encode_times)/len(encode_times))
    print("Decode Average", sum(decode_times)/len(decode_times))


# print(ntru_end_to_end('teststring',n=167,p=3,q=128))



# text_file = open("lorem_ipsum_test.txt", "r").read()

# print(ntru_end_to_end(text_file))




# lorem_ipsum = open("lorem_ipsum_test.txt", "r").read()

# # ntru_output = ntru_end_to_end_ternary(lorem_ipsum, n=167, p=3, q=128)

# # print(ntru_output == lorem_ipsum)

# base_two = []

# for i in range(10):
#     tic = time.perf_counter()
#     ntru_output = ntru_end_to_end(lorem_ipsum, n=401, p=3, q=2048)
#     toc = time.perf_counter()
#     base_two.append(toc-tic)
#     print(toc-tic)

# print("BASE 2 TIMES", base_two)
# print("average: ", sum(base_two)/len(base_two))


# base_three = []

# for i in range(10):
#     tic = time.perf_counter()
#     ntru_output = ntru_end_to_end_ternary(lorem_ipsum, n=401, p=3, q=2048)
#     toc = time.perf_counter()
#     base_three.append(toc-tic)
#     print(toc-tic)

# print("BASE 3 TIMES", base_three)
# print("average: ", sum(base_three)/len(base_three))

# n_values = [401,439,593,743]

# for n in n_values:
#     encrypt_time = []
#     decrypt_time = []
#     keys = key_gen(n,p=3,q=2048)
#     print("Key Generated")
#     for i in range(20):
#         tic = time.perf_counter()
#         encrypted_message = encrypt("The quick brown fox",keys['h'],n,3,2048)[0]
#         toc = time.perf_counter()
#         decrypted_message = decrypt(encrypted_message,keys['f'],keys['f_p'],n,3,2048)
#         toc_two = time.perf_counter()
#         # print(toc-tic)
#         # print(toc_two-toc)
#         encrypt_time.append(toc-tic)
#         decrypt_time.append(toc_two-toc)

#     print(n)
#     print(encrypt_time)
#     print(sum(encrypt_time)/len(encrypt_time))
#     print("Decrypt time")
#     print(decrypt_time)
#     print(sum(decrypt_time)/len(decrypt_time))


# for n in n_values:
#     decrypt_time = []
#     keys = key_gen(n,p=3,q=2048)
#     encrypted_message = encrypt("The quick brown fox jumps over the lazy dog",keys['h'],n,3,2048)
#     print("Key Generated")
#     for i in range(10):
#         tic = time.perf_counter()
#         decrypted_message = decrypt(encrypted_message,keys['f'],keys['f_p'],n,3,2048)
#         toc = time.perf_counter()
#         print(toc-tic)
#         decrypt_time.append(toc-tic)

#     print(decrypt_time)
#     print(sum(decrypt_time)/len(decrypt_time))


# print(ntru_with_parity("test", n=401))


# ntru_n_values = [401,439,593,743]
# total_attempts = 0
# attempt_array = []

# for i in range(0,50):
#     print("here")
#     ntru_instance = ntru(401,3,2048)
#     attempt = ntru_instance.f_gen(20)
#     total_attempts += attempt
#     attempt_array.append(attempt)


# print(attempt_array)

# print("average attempts", total_attempts/50)



# print(key_gen_helper())

# print(ntru_aes_package())

# # print(ntru_end_to_end("super duper long string just to see whether this can be decoded a;sjdf;ajsdfkl;jals;dfjl;asjdf;ajsdl;fja;lksdfj"))


# # value_list = [87,503,347,251,401]

# # value_list = [401,251]

# # total_unsucc = 0 
# # for j in value_list:
# #   successful_cnt = 0
# #   unsuccessful_cnt = 0
# #   for i in range(0,50):
# #       alpha = 1
# #       beta = 1
# #       gamma = 1
#       tetta = 1
#       eta = 1

#       random_ntru_instance = 1
#       original_m = Poly(-1 + random_ntru_instance*x + random_ntru_instance*x**2 + x**3 + x**4 + x**9 + x**10 + x**166,x).set_domain(ZZ)


#       ntru_instance = ntru(j,3,2048)
#       ntru_instance.key_gen()
#       encrypted_m = ntru_instance.encrypt(original_m)
#       decrypted_m = ntru_instance.decrypt(encrypted_m)

#       if(decrypted_m == original_m):
#           print("SUCCESSFUL DECRYPTION")
#           successful_cnt += 1
#       else:
#           print("FAILED DECRYPTION")
#           unsuccessful_cnt += 1


#   print("Successful: ", successful_cnt)

#   print("Unsuccessful: ", unsuccessful_cnt)
#   total_unsucc += unsuccessful_cnt


