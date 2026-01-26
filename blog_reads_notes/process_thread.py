import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import requests

def compute_sum_squares(n):
    return sum(i*i for i in range(n))

def fetch_url(url):
    response = requests.get(url)
    return response.status_code

if __name__ == "__main__":
    #### IO Bound task
    urls = [
    "https://httpbin.org/delay/3",
    "https://httpbin.org/delay/3",
    "https://httpbin.org/delay/3",
    ]

    #### Compute Bound task
    n = [10_000_000, 20_000_000, 30_000_000, 40_000_000]

    ### Single threaded
    # start = time.time()
    # results = [fetch_url(url) for url in urls]
    # results = [compute_sum_squares(i) for i in n]
    # end = time.time()
    # print(f"Single threaded: {end - start}")

    # ### Multithreading using ThreadPoolExecutor
    # start = time.time()
    # with ThreadPoolExecutor as executor:
    #     results = list(executor.map(fetch_url, urls))
    #     results = list(executor.map(compute_sum_squares, n))
    # end = time.time()
    # print(f"Multithreaded using ThreadPoolExecutor: {end - start}")

    ### MultiProcess using ProcessPoolExecutor
    # start = time.time()
    # with ProcessPoolExecutor as executor:
    #     results = list(executor.map(fetch_url, urls))
    #     results = list(executor.map(compute_sum_squares, n))
    # end = time.time()
    # print(f"MultiProcess using ProcessPoolExecutor: {end - start}")
