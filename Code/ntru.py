#Main math imports
import numpy as np
from random import randint
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

    print("Bit array", bit_array)

    return bit_array



def bits_to_bytes(bit_array):
    byte_value = 0
    two_power = 1

    for bit in bit_array[::-1]:
        # print("bit", bit)
        byte_value += two_power * bit
        two_power = two_power * 2


    if byte_value > 255:
        raise ValueError("Max Byte val: ", max(byte_array), " not in valid range")

    print("Byte Value", byte_value)

    return byte_value


def bit_padding(bit_array):
    excess = len(bit_array) % 8

    if excess == 0:
        return bit_array

    else:
        bit_array = [0]*(8-excess) + bit_array
        return bit_array


def string_decode(bit_array):
    output_string = ""
    for i in range(0,int(len(bit_array)/8)):
        output_string += bytes([bits_to_bytes(bit_array[i*8:(i+1)*8])]).decode("utf-8")

    return output_string
        

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
            r_factors += randint(-1,1) * x**i

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
            print(attempts)
            f_factors = 0
            for i in range(0,self.n):
                f_factors += randint(-1,1) * x**i


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

        self.f = self.f_gen(20) #can change max attempts here, set to 20 by default as this was sufficient in early testing

        self.h = (((self.p * self.f_q).trunc(self.q) * self.g).trunc(self.q) % self.r_space).trunc(self.q)

        print("Public Key: ", self.h)


    def encrypt(self,m):
        print("Original message", m)

        r = self.r_gen()

        r_h = (r * self.h).trunc(self.q)

        e = ((r_h + m) % self.r_space).trunc(self.q)

        return e


    def decrypt(self,encrypted_m):

        a = ((self.f * encrypted_m) % self.r_space).trunc(self.q)
        b = a.trunc(self.p)

        c = ((self.f_p * b) % self.r_space).trunc(self.p)

        return c



test = ntru(167,3,128)
test.key_gen()

string = "liluzivertsantandave"

encoded_string = string.encode("utf-8")

byte_list = list(encoded_string)

bit_list = bytes_to_bits(byte_list)

original_m = Poly(bit_list,x).set_domain(ZZ)
print(original_m)

encrypted_m = test.encrypt(original_m)
decrypted_m = test.decrypt(encrypted_m)

print("Decrypted polynomial", decrypted_m)

coeffs = bit_padding(decrypted_m.all_coeffs())

print("coeffs", coeffs)

print(type(coeffs))

# print("byte coeffs", bits_to_bytes(coeffs))

decoded_string = string_decode(coeffs)


# decoded_string = bytes([bits_to_bytes(coeffs)]).decode("utf-8")

print("Decoded string", decoded_string)



# value_list = [87,503,347,251,167]

value_list = [167,251]

# total_unsucc = 0 
# for j in value_list:
#   successful_cnt = 0
#   unsuccessful_cnt = 0
#   for i in range(0,50):
#       alpha = 1
#       beta = 1
#       gamma = 1
#       tetta = 1
#       eta = 1

#       random_test = 1
#       original_m = Poly(-1 + random_test*x + random_test*x**2 + x**3 + x**4 + x**9 + x**10 + x**166,x).set_domain(ZZ)


#       test = ntru(j,3,128)
#       test.key_gen()
#       encrypted_m = test.encrypt(original_m)
#       decrypted_m = test.decrypt(encrypted_m)

#       if(decrypted_m == original_m):
#           print("SUCCESSFUL DECRYPTION")
#           successful_cnt += 1
#       else:
#           print("FAILED DECRYPTION")
#           unsuccessful_cnt += 1


#   print("Successful: ", successful_cnt)

#   print("Unsuccessful: ", unsuccessful_cnt)
#   total_unsucc += unsuccessful_cnt


