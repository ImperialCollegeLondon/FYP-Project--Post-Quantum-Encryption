import random
import sympy
#Importing maths libraries for generating random numbers and prime checking

n = 15 #Number to be factorised


attempts_required = 0

while True:
    attempts_required += 1
    x = random.randint(2,n-2) #Initialises a random variable where 1 < x < n-1 

    range_register = [i for i in range(n+1)] #Creates an array of numbers from 0 to n inclusive
    remainder_register = []

    for num in range_register:
        remainder_register.append(x**num % n)

    print(range_register)
    print(remainder_register)

    sequence_length_len = len(set(remainder_register)) #Gets the length of the unique sequence. e.g. [1, 2, 4, 1, 2, 4] returns 3.

    possible_factor = x^(sequence_length_len//2) - 1
    complimentary_factor = int(n / possible_factor)

    if n % possible_factor != 0:
        continue
    elif sympy.isprime(possible_factor) and sympy.isprime(complimentary_factor): #Exit condition, when we find two primes that multiply to give n
        break
            

print('First prime: ', possible_factor)
print('Second prime: ', complimentary_factor)
print('Attempts required: ', attempts_required)

