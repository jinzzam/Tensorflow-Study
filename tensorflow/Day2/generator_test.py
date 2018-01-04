import math
def is_prime(number):
    if number <= 1:
        return False
    if number == 2:
        return True
    if number % 2 == 0:
        return False
    for div in range(3, int(math.sqrt(number) + 1), 2):
        if number % div == 0:
            return False
    return True

def get_primes(number):     # yield를 만날 때 까지 반복  : break 와 비슷
    while True:
        if is_prime(number):
            yield number    # yield를 만나고 그 후 number 값은 freeze
        number += 1

prime_iterator = get_primes(1)      # iterator : 반복하다

for _ in range(100) :       # 100번 반복해서 돌아라
    next_prime_number = next(prime_iterator)    # next를 호출하는 순간 get_primes 함수가 호출되어 실행됨
    print(next_prime_number)