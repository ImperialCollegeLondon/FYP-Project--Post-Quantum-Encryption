#Main math imports
import os
from base64 import b64encode
import numpy as np
import random
import string
import math



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



def bits_to_bytes(bit_array):
    byte_value = 0
    two_power = 1

    for bit in bit_array[::-1]:
        byte_value += two_power * bit
        two_power = two_power * 2


    if byte_value > 255:
        raise ValueError("Max Byte val: ", max(byte_array), " not in valid range")

    return byte_value


def bit_padding(bit_array,num_of_bits):
    excess = len(bit_array) % num_of_bits

    if excess == 0:
        return bit_array

    else:
        bit_array = [0]*(num_of_bits-excess) + bit_array
        return bit_array


def string_decode(bit_array):
    output_string = ""
    for i in range(0,int(len(bit_array)/8)):
        output_string += bytes([bits_to_bytes(bit_array[i*8:(i+1)*8])]).decode("utf-8")

    return output_string


def parity_encode(bit_array):
    one_count = sum(bit_array)
    parity_bit = one_count % 2
    bit_array.append(parity_bit)
    return bit_array


def parity_checker(bit_array,parity_bit):


    if sum(bit_array) % 2 == parity_bit:
        return True

    else:
        return False

        

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

        self.n = n
        self.p = p
        self.q = q
        self.r_space = Poly(x**n - 1, x).set_domain(ZZ)


    def g_gen(self):
        num_ones = int(math.sqrt(self.q))
        num_zeros = self.n - 2*num_ones

        g_factors = np.random.permutation(np.concatenate((np.zeros(num_zeros), np.ones(num_ones), -np.ones(num_ones))))
        
        return Poly(g_factors,x).set_domain(ZZ)

    def r_gen(self):
        r_factors = 0
        for i in range(0,self.n):
            r_factors += random.randint(-1,1) * x**i

        r = Poly(r_factors,x).set_domain(ZZ)

        return r


    def f_invert(self,f,mod_param):
        if math.log(mod_param,2) == int(math.log(mod_param,2)): 
            f_inv = invert(f,self.r_space, domain=GF(2,symmetric = False))

            log_val = int(math.log(mod_param, 2))

            for i in range(1, log_val):
                f_inv = ((2 * f_inv - f * f_inv ** 2) % self.r_space).trunc(mod_param)
            return f_inv
        else:
            return invert(f,self.r_space,domain=GF(mod_param))


    def f_gen(self,max_attempts):
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

        self.g = self.g_gen()

        self.f = self.f_gen(20) #can change max attempts here, set to 20 by default as this was sufficient in early ntru_instanceing

        self.h = (((self.p * self.f_q).trunc(self.q) * self.g).trunc(self.q) % self.r_space).trunc(self.q)


    def encrypt(self,m):

        r = self.r_gen()

        r_h = (r * self.h).trunc(self.q)

        e = ((r_h + m) % self.r_space).trunc(self.q)

        return e


    def decrypt(self,encrypted_m):

        a = ((self.f * encrypted_m) % self.r_space).trunc(self.q)
        b = a.trunc(self.p)

        c = ((self.f_p * b) % self.r_space).trunc(self.p)

        return c



def ntru_end_to_end(message_string, n = 167, p = 3 , q = 128, detailed_stats = False):
    
    ntru_instance = ntru(n,p,q)
    ntru_instance.key_gen()
    full_string_length = len(message_string)

    # message_string = "test"

    decoded_full_string = ""

    max_chars = int(math.floor(n/8))

    if len(message_string) % max_chars == 0:
        splits = int(math.floor(full_string_length/max_chars)) #define to be 8 here because we're just using UTF-8 256. 

    else:
        splits = splits = int(math.floor(full_string_length/max_chars)) + 1

    for i in range(0,splits):

        # partial_msg_string = message_string[(i*max_chars) : (i+1)*max_chars]
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



# text_file = open("lorem_ipsum_test.txt", "r").read()

# print(ntru_end_to_end(text_file))


def ntru_aes_package(aes_size=256,n=167,p=3,q=128, detailed_stats = False):

    valid_aes_sizes = [128,192,256]
    if aes_size not in valid_aes_sizes:
        raise ValueError("AES size has to be 128, 192 or 256 bits")

    aes_key = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits + string.punctuation) for _ in range(int(aes_size/8)))

    aes_key_byte_rep = aes_key.encode("utf-8")

    print(aes_key)

    return (ntru_end_to_end(aes_key, n, p, q, detailed_stats),aes_key_byte_rep)
    
    

def key_gen_helper(n=167,p=3,q=128):
    ntru_instance = ntru(n,p,q)
    ntru_instance.key_gen()

    return {"g":ntru_instance.g, "f": ntru_instance.f, "h": ntru_instance.h, "f_p": ntru_instance.f_p, "f_q": ntru_instance.f_q}



def ntru_end_to_end_ternary(message_string, n = 167, p = 3 , q = 128, detailed_stats = False):
    
    ntru_instance = ntru(n,p,q)
    ntru_instance.key_gen()
    full_string_length = len(message_string)

    # message_string = "test"

    decoded_full_string = ""

    max_chars = int(math.floor(n/8))

    if len(message_string) % max_chars == 0:
        splits = int(math.floor(full_string_length/max_chars)) #define to be 8 here because we're just using UTF-8 256. 

    else:
        splits = splits = int(math.floor(full_string_length/max_chars)) + 1

    for i in range(0,splits):

        # partial_msg_string = message_string[(i*max_chars) : (i+1)*max_chars]
        partial_msg_string = message_string[(i*max_chars) : min(len(message_string),(i+1)*max_chars)]
        
        encoded_string = partial_msg_string.encode("utf-8")

        byte_list = list(encoded_string)

        bit_list = bytes_to_bits(byte_list)

        original_m = Poly(bit_list,x).set_domain(ZZ)

        encrypted_m = ntru_instance.encrypt(original_m)
        decrypted_m = ntru_instance.decrypt(encrypted_m)

        coeffs = bit_padding(decrypted_m.all_coeffs(),6)

        decoded_string = string_decode(coeffs)

        decoded_full_string += decoded_string


    if detailed_stats:
        detailed_stats_dict = {"f": ntru_instance.f, "g": ntru_instance.g, "f_p": ntru_instance.f_p, "f_q": ntru_instance.f_q}
        return(detailed_stats_dict,decoded_full_string)
    
    else:
        return(decoded_full_string)



def ntru_with_parity(message_string, n = 167, p = 3 , q = 128, detailed_stats = False):
    
    ntru_instance = ntru(n,p,q)
    ntru_instance.key_gen()
    full_string_length = len(message_string)

    # message_string = "test"

    decoded_full_string = ""

    max_chars = int(math.floor(n/8))

    if len(message_string) % max_chars == 0:
        splits = int(math.floor(full_string_length/max_chars)) #define to be 8 here because we're just using UTF-8 256. 

    else:
        splits = splits = int(math.floor(full_string_length/max_chars)) + 1

    for i in range(0,splits):

        # partial_msg_string = message_string[(i*max_chars) : (i+1)*max_chars]
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


print(ntru_with_parity("test", n=167))







# print(key_gen_helper())

# print(ntru_aes_package())

# # print(ntru_end_to_end("super duper long string just to see whether this can be decoded a;sjdf;ajsdfkl;jals;dfjl;asjdf;ajsdl;fja;lksdfj"))


# # value_list = [87,503,347,251,167]

# # value_list = [167,251]

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


#       ntru_instance = ntru(j,3,128)
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


