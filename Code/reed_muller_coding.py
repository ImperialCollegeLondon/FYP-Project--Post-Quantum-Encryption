import numpy as np
import math
import itertools

def decimal_to_binary_comb(number,bin_len):

    """
    Name:        decimal_to_binary_comb

    Description: Converts decimal number to binary representation, replacing 0 with -1
                 Used to find permutations of a particular row
    Arguments:   - number: Decimal number to be converted
                 - bin_len: Length of representation as output 

    Returns:     - bin_list: Binary representation list of number, with 0 replaced as -1
    """



    bin_lst = [int(bit) for bit in np.binary_repr(number,width=bin_len)]
    bin_lst = [-1 if x==0 else x for x in bin_lst]
    return bin_lst



def matrix_gen(r,m):

    """
    Name:        matrix_gen

    Description: Generates Reed-Muller Generator Matrix and associated variables
                 required for encode and decode functions
    Arguments:   - r: Reed-Muller parameter, output block size.
                 - m: Reed-Muller parameter, input mesage size.

    Returns:     - reed_muller_gen_matrix: Reed-Muller Generator Matrix
                 - reed_muller_inverse_matrix: Reed-Muller Inverse Matrix, used for decode
                 - val_list: List used for decoding, contains row count info.
    """
    row_length = 2**m
    ones = math.floor(row_length/2)
    zeros = math.floor(row_length/2)

    reed_muller_gen_matrix = np.array(row_length*[1])
    reed_muller_inverse_matrix = np.array(row_length*[1])

    val_list = []
    counter = 1

    while ones >= 1:
        val_list.append(tuple([counter]))
        counter += 1
        partial_row = ones*[1] + zeros*[0]
        full_row = np.array(int(row_length/len(partial_row)) * partial_row)
        reed_muller_inverse_matrix = np.vstack([reed_muller_inverse_matrix,1-full_row])
        reed_muller_gen_matrix = np.vstack([reed_muller_gen_matrix,full_row])
        zeros -= math.ceil(zeros/2)
        ones -= math.ceil(ones/2)

    perm_string = [i for i in range(1,m+1)]

    one_array = np.array(row_length*[1])

    for i in range(2,r+1):
        for j in itertools.combinations(perm_string,r=i):
            full_row = one_array
            val_list.append(j)
            for k in j:
                full_row = np.multiply(full_row,reed_muller_gen_matrix[int(k)])

            reed_muller_gen_matrix = np.vstack([reed_muller_gen_matrix,full_row])


    return (reed_muller_gen_matrix,reed_muller_inverse_matrix,val_list)


def encode(message,r,m,reed_muller_gen_matrix):
    """
    Name:        encode

    Description: Encodes message bit stream into Reed-Muller bit stream.
    Arguments:   - r: Reed-Muller parameter, output block size.
                 - m: Reed-Muller parameter, input mesage size.
                 - reed_muller_gen_matrix: Generator matrix, used to encode message.
                   Derived from matrix_gen function.

    Returns:     - encoded_transmisison: Message bit stream encoded as Reed-Muller code.
    """
    encoded_transmission = np.dot(message,reed_muller_gen_matrix) % 2

    return encoded_transmission


def decode(encoded_transmission,r,m,reed_muller_gen_matrix,reed_muller_inverse_matrix,val_list):
    """
    Name:        encode

    Description: Encodes message bit stream into Reed-Muller bit stream.
    Arguments:   - encoded_transmisison: Message bit stream encoded as Reed-Muller code.
                 - r: Reed-Muller parameter, output block size.
                 - m: Reed-Muller parameter, input mesage size.
                 - reed_muller_gen_matrix: Generator matrix, used to decode message.
                   Derived from matrix_gen function.
                 - reed_muller_inverse_matrix: Inverse Reed-Muller generator matrix
                 - val_list: Contains row count info 

    Returns:     - decoded_message: Returns the original message bit stream. 
    """
    row_length = 2**m
    perm_string = [i for i in range(1,m+1)]

    u = encoded_transmission

    decoded_message = []

    for degree in range(r,-1,-1):
        collected_rows = np.empty((0,row_length), int)

        if degree == 0:
            start_row = 0
            end_row = -1

        else:
            start_row = sum(math.comb(m,i) for i in range(0,degree+1)) - 2
            end_row = sum(math.comb(m,i) for i in range(0,degree)) - 2

        degree_stored = []

        for row in range(start_row,end_row,-1):
            collected_rows = np.vstack([reed_muller_gen_matrix[row+1],collected_rows])

            if degree == 0:
                vectors = perm_string
            
            else:
                vectors = list(np.setdiff1d(perm_string, val_list[row]))

            one_cnt = 0
            zero_cnt = 0

            for i in range(0,2**len(vectors)): #performs for all numbers, e.g. 0,1,2,3
                combined_arr = np.array([1]*row_length)

                bin_num = decimal_to_binary_comb(i, len(vectors))

                combinations = [a*b for a,b in zip(bin_num,vectors)]

                for vector in combinations: #performs within that combination e.g. (1,2) then it will multiply and find the combined array for it accordingly.
                    if abs(vector) != vector: #i.e. if negative number, then means inverse is to be applied
                        combined_arr = np.multiply(combined_arr,reed_muller_inverse_matrix[abs(vector)])
                    else:
                        combined_arr = np.multiply(combined_arr,reed_muller_gen_matrix[vector])

                if np.dot(u,combined_arr) % 2 == 0:
                    zero_cnt += 1

                else:
                    one_cnt += 1

            if one_cnt > zero_cnt:
                degree_stored = [1] + degree_stored
            
            elif zero_cnt > one_cnt:
                degree_stored = [0] + degree_stored

            else:
                raise RuntimeError("Corruption cannot be corrected. Please resend the message")
        
        s = (np.dot(np.array([degree_stored]),collected_rows) % 2 + u) % 2


        u = s

        decoded_message = degree_stored + decoded_message

    return decoded_message