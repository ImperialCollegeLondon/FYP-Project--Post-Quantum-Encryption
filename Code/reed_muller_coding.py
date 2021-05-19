import numpy as np
import math
import itertools

def decimal_to_binary_comb(number,bin_len):
    bin_lst = [int(bit) for bit in np.binary_repr(number,width=bin_len)]
    bin_lst = [-1 if x==0 else x for x in bin_lst]
    return bin_lst

r = 2 #the degree of the polynomial/factors later.
m = 4 #also specifies the width/projected space of the new message.


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

# print(reed_muller_gen_matrix)
# print("inverse", reed_muller_inverse_matrix)

perm_string = [i for i in range(1,m+1)]

one_array = np.array(row_length*[1])


for i in range(2,r+1):
    for j in itertools.combinations(perm_string,r=i):
        full_row = one_array
        val_list.append(j)
        for k in j:
            full_row = np.multiply(full_row,reed_muller_gen_matrix[int(k)])

        reed_muller_gen_matrix = np.vstack([reed_muller_gen_matrix,full_row])


# print(reed_muller_gen_matrix)


# test_matrix = np.array([0,1,1,1,0,1,0])
# test_matrix = np.array([1,0,1,1,1,1,1,1,1,1,1,0,1,1,0])

test_matrix = np.array([1,1,1,1,0,1,0,0,1,1,0])

print(val_list)

encoded_transmission = np.dot(test_matrix,reed_muller_gen_matrix) % 2

print(encoded_transmission)

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


print("Decoded Message: ", decoded_message)








# reed_muller_matrix = np.array(np.mat(reed_muller_txt))

# new_row = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])

# reed_muller_matrix = np.vstack([reed_muller_matrix, new_row])

# print(reed_muller_matrix)

# # test = """0 1 1 0 1 0 0 1 0 1 0"""

# # test_matrix = np.matrix(test)

# # print(np.dot(test_matrix,reed_muller_matrix) % 2)


