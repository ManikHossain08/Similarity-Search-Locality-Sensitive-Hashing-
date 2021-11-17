from pyspark.sql import SparkSession
from math import sqrt,log
from scipy.special import lambertw
from random import randint
from pretty_print_dict import pretty_print_dict as ppd
from pretty_print_bands import pretty_print_bands as ppb
import random
# Dask imports
import dask.bag as db
import dask.dataframe as df


all_states = ["ab", "ak", "ar", "az", "ca", "co", "ct", "de", "dc",
              "fl", "ga", "hi", "id", "il", "in", "ia", "ks", "ky", "la",
              "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv",
              "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa",
              "pr", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "vi",
              "wa", "wv", "wi", "wy", "al", "bc", "mb", "nb", "lb", "nf",
              "nt", "ns", "nu", "on", "qc", "sk", "yt", "dengl", "fraspm"]


def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark


# def toCSVLineRDD(rdd):
#     """This function is used by toCSVLine to convert an RDD into a CSV string
#
#     """
#     a = rdd.map(lambda row: ",".join([str(elt) for elt in row]))\
#            .reduce(lambda x, y: "\n".join([x, y]))
#     return a + "\n"
#
#
# def toCSVLine(data):
#     """This function convert an RDD or a DataFrame into a CSV string
#
#     """
#     if isinstance(data, RDD):
#         return toCSVLineRDD(data)
#     elif isinstance(data, DataFrame):
#         return toCSVLineRDD(data.rdd)
#     return None

def data_preparation0(data_file,sc):
    states = sc.parallelize(all_states)
    states = states.map(lambda row: dict({row: 0})).reduce(lambda x, y: {**x, **y})

    rdd = sc.textFile(data_file) \
        .map(lambda row: row.split(",")) \
        .flatMap(lambda l: [(l[0], dict({l[i]: 1})) for i in range(1, len(l))]) \
        .reduceByKey(lambda x, y: {**x, **y}) \
        .map(lambda x: (x[0], {**states, **x[1]}))
    return rdd

def data_preparation(data_file, key, state):
    """Our implementation of LSH will be based on RDDs. As in the clustering
    part of LA3, we will represent each state in the dataset as a dictionary of
    boolean values with an extra key to store the state name.
    We call this dictionary 'state dictionary'.

    Task 1 : Write a script that
             1) Creates an RDD in which every element is a state dictionary
                with the following keys and values:

                    Key     |         Value
                ---------------------------------------------
                    name    | abbreviation of the state
                    <plant> | 1 if <plant> occurs, 0 otherwise

             2) Returns the value associated with key
                <key> in the dictionary corresponding to state <state>

    *** Note: Dask may be used instead of Spark.

    Keyword arguments:
    data_file -- csv file of plant name/states tuples (e.g. ./data/plants.data)
    key -- plant name
    state -- state abbreviation (see: all_states)
    """
    sc = init_spark().sparkContext
    rdd=data_preparation0(data_file,sc)
    rdd=rdd.filter(lambda x:x[0]==key).map(lambda x:x[1][state]).collect()[0]
    return rdd

def isPrime(x):
    for i in range(2,int(sqrt(x))+1):
        if (x%i)==0:
            return False
    return True
def primes(n, c):
    """To create signatures we need hash functions (see next task). To create
    hash functions,we need prime numbers.

    Task 2: Write a script that returns the list of n consecutive prime numbers
    greater or equal to c. A simple way to test if an integer x is prime is to
    check that x is not a multiple of any integer lower or equal than sqrt(x).

    Keyword arguments:
    n -- integer representing the number of consecutive prime numbers
    c -- minimum prime number value
    """
    sc = init_spark().sparkContext
    numbers = sc.parallelize(i for i in range(c,1000000))
    numbers=numbers.map(lambda x:(x,isPrime(x))).filter(lambda x:x[1]==True).map(lambda x:x[0]).take(n)
    return numbers

def hash_plants(s, m, p, x):
    """We will generate hash functions of the form h(x) = (ax+b) % p, where a
    and b are random numbers and p is a prime number.

    Task 3: Write a function that takes a pair of integers (m, p) and returns
    a hash function h(x)=(ax+b)%p where a and b are random integers chosen
    uniformly between 1 and m, using Python's random.randint. Write a script
    that:
        1. initializes the random seed from <seed>,
        2. generates a hash function h from <m> and <p>,
        3. returns the value of h(x).

    Keyword arguments:
    s -- value to initialize random seed from
    m -- maximum value of random integers
    p -- prime number
    x -- value to be hashed
    """
    random.seed(s)
    a=random.randint(1,m)
    b=random.randint(1, m)
    h=(a*x+b)%p
    return h

def hash_list(s, m, n, i, x):
    """We will generate "good" hash functions using the generator in 3 and
    the prime numbers in 2.

    Task 4: Write a script that:
        1) creates a list of <n> hash functions where the ith hash function is
           obtained using the generator in 3, defining <p> as the ith prime
           number larger than <m> (<p> being obtained as in 1),
        2) prints the value of h_i(x), where h_i is the ith hash function in
           the list (starting at 0). The random seed must be initialized from
           <seed>.

    Keyword arguments:
    s -- seed to intialize random number generator
    m -- max value of hash random integers
    n -- number of hash functions to generate
    i -- index of hash function to use
    x -- value to hash
    """
    result=[]
    numbers=primes(n,m)
    random.seed(s)
    for p in numbers:
        result.append((random.randint(1, m) * x + (random.randint(1, m))) % p)
    return result[i]

def signatures(datafile, seed, n, state):
    """We will now compute the min-hash signature matrix of the states.

    Task 5: Write a function that takes build a signature of size n for a
            given state.

    1. Create the RDD of state dictionaries as in data_preparation.
    2. Generate `n` hash functions as done before. Use the number of line in
       datafile for the value of m.
    3. Sort the plant dictionary by key (alphabetical order) such that the
       ordering corresponds to a row index (starting at 0).
       Note: the plant dictionary, by default, contains the state name.
       Disregard this key-value pair when assigning indices to the plants.
    4. Build the signature array of size `n` where signature[i] is the minimum
       value of the i-th hash function applied to the index of every plant that
       appears in the given state.


    Apply this function to the RDD of dictionary states to create a signature
    "matrix", in fact an RDD containing state signatures represented as
    dictionaries. Write a script that returns the string output of the RDD
    element corresponding to state '' using function pretty_print_dict
    (provided in answers).

    The random seed used to generate the hash function must be initialized from
    <seed>, as previously.

    ***Note: Dask may be used instead of Spark.

    Keyword arguments:
    datafile -- the input filename
    seed -- seed to initialize random int generator
    n -- number of hash functions to generate
    state -- state abbreviation
    """
    sc = init_spark().sparkContext
    rdd = data_preparation0(datafile, sc).sortByKey()
    m = rdd.count()
    rdd=rdd.zipWithIndex().filter(lambda x:x[0][1][state]==1)
    plantIndex=rdd.map(lambda x:x[1])
    numbers = primes(n, m)
    result={}
    random.seed(seed)
    for index,num in enumerate(numbers):
        a=random.randint(1, m)
        b=random.randint(1, m)
        r=plantIndex.map(lambda i:(a * i +b ) % num).min()
        result[index]=r
    return result

def hash_band(datafile, seed, state, n, b, n_r):
    """We will now hash the signature matrix in bands. All signature vectors,
    that is, state signatures contained in the RDD computed in the previous
    question, can be hashed independently. Here we compute the hash of a band
    of a signature vector.

    Task 6: Write a script that, given the signature dictionary of state <state>
    computed from <n> hash functions (as defined in the previous task),
    a particular band <b> and a number of rows <n_r>:

    1. Generate the signature dictionary for <state>.
    2. Select the sub-dictionary of the signature with indexes between
       [b*n_r, (b+1)*n_r[.
    3. Turn this sub-dictionary into a string.
    4. Hash the string using the hash built-in function of python.

    The random seed must be initialized from <seed>, as previously.

    Keyword arguments:
    datafile --  the input filename
    seed -- seed to initialize random int generator
    state -- state to filter by
    n -- number of hash functions to generate
    b -- the band index
    n_r -- the number of rows
    """
    state_dict=signatures(datafile, seed, n, state)
    sub_dict={}
    for key in state_dict:
        if b*n_r<=key and key<(b+1)*n_r:
            sub_dict[key]=state_dict[key]
    sub_dict=str(sub_dict)
    return hash(sub_dict)

def hash_bands(data_file, seed, n_b, n_r):
    """We will now hash the complete signature matrix

    Task 7: Write a script that, given an RDD of state signature dictionaries
    constructed from n=<n_b>*<n_r> hash functions (as in 5), a number of bands
    <n_b> and a number of rows <n_r>:

    1. maps each RDD element (using flatMap) to a list of ((b, hash),
       state_name) tuples where hash is the hash of the signature vector of
       state state_name in band b as defined in 6. Note: it is not a triple, it
       is a pair.
    2. groups the resulting RDD by key: states that hash to the same bucket for
       band b will appear together.
    3. returns the string output of the buckets with more than 2 elements
       using the function in pretty_print_bands.py.

    That's it, you have printed the similar items, in O(n)!

    Keyword arguments:
    datafile -- the input filename
    seed -- the seed to initialize the random int generator
    n_b -- the number of bands
    n_r -- the number of rows in a given band
    """
    sc = init_spark().sparkContext
    rdd = data_preparation0(data_file, sc).sortByKey()
    m = rdd.count()
    rdd = rdd.zipWithIndex()
    numbers = primes(n_b * n_r, m)
    rdd=rdd.flatMap(lambda x:[(x[1],key,x[0][1][key]) for key in x[0][1]]).filter(lambda x:x[2]==1)
    plantIndex = rdd.map(lambda x: (x[1],[x[0]])).reduceByKey(lambda x,y:x+y) #(state,[IndexOfPlants])
    result = sc.parallelize([])
    random.seed(seed)
    for index, num in enumerate(numbers):
        a = random.randint(1, m)
        b = random.randint(1, m)
        r = plantIndex.flatMap(lambda i: [(i[0],[(index,(a * j + b) % num)]) for j in i[1]]) \
            .reduceByKey(lambda x,y:x+y) \
            .map(lambda x:(x[0],min(x[1],key=lambda x:x[1]))) \
            .map(lambda x:(x[0],{x[1][0]:x[1][1]}))
        result=result.union(r)
        result=result.reduceByKey(lambda x,y:{**x,**y})
    result=result.flatMap(lambda x:[(x[0],{key:x[1][key]}) for key in x[1]])
    finalResult = sc.parallelize([])
    for y in range(0, n_b):
        result1=result.filter(lambda x:(y * n_r <= list(x[1].keys())[0] and list(x[1].keys())[0] < (y + 1) * n_r)) \
            .reduceByKey(lambda x, y: {**x, **y}) \
            .map(lambda x:((y,hash(str(x[1]))),x[0]))
        finalResult=finalResult.union(result1)
    finalResult=finalResult.groupByKey().filter(lambda x:len(x[1])>1)
    return ppb(finalResult)

def get_b_and_r(n, s):
    """The script written for the previous task takes <n_b> and <n_r> as
    parameters while a similarity threshold <s> would be more useful.

    Task: Write a script that prints the number of bands <b> and rows <r> to be
    used with a number <n> of hash functions to find the similar items for a
    given similarity threshold <s>. Your script should also print <n_actual>
    and <s_actual>, the actual values of <n> and <s> that will be used, which
    may differ from <n> and <s> due to rounding issues. Printing format is
    found in tests/test-get-b-and-r.txt

    Use the following relations:

     - r=n/b
     - s=(1/b)^(1/r)

    Hint: Johann Heinrich Lambert (1728-1777) was a Swiss mathematician

    Keywords arguments:
    n -- the number of hash functions
    s -- the similarity threshold

    b.S^r=1
    n=r/S^r
    n=r.e^(-r.ln(s))
    -n.ln(s)=-r.ln(s).e^(-r.ln(s))
    W(-n.ln(s))=-r.ln(s)
    """

    x=(-n)*log(s)
    r=lambertw(x)/(-log(s))
    r=int(r.real)+1
    b=int(n/r)
    real_n=r*b
    s_real=(1/b)**(1/r)
    result="b="+str(b)+"\nr="+str(r)+"\nn_real="+str(real_n)+"\ns_real="+str(s_real)+"\n"
    return result