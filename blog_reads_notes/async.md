## Async/Await in Python

### TL;DR Notes

- We can use async/await only for (almost) independent function calls such as IO, processing on different files etc; If there's dependence i.e one function needs , this won't work
- **Coroutine** - A function which has `async` keyword (mandatory) and `await` keyword (not mandatory) inside it
- **Event Loop** - An infinite loop which keeps checking for any async tasks which are awaiting for resources, this is like a scheduler, manager, traffic controller which keeps track of all async tasks and allocates resources to them as and when they are free. Under the hood, this is a queue which manage tasks
- Do `await` on **tasks** (create tasks on coroutines using `asyncio.create_task()` and do `await asyncio.gather()`) for best event loop efficiency. awaiting on coroutines is not ideal as it will simply step inside it
- **Example 2** - async on coroutines (read as `await coroutine`) will directly step inside the function and runs/utilizes the resources. So, if there's any sleep (real world IO) wait times inside coroutines, this will block the resources for event loop and thus works same as synchronous
- **Example 3** - If there are no `await` inside the coroutines (which is valid), even though all tasks are created and fired at once i.e async on tasks, the resources will still be throttled by one or other coroutine because we didn't say where the loop is allowed to interrupt your function to run another task while waiting on the data (IO) or result.
- TODOs: Blocking the event loop, yielding the event loop ...
---

## Example 1: Proper Async/Await Usage

Proper example of how async await works:
- `main()` is kept on event loop control by `asyncio.run()`
- All `async` tasks are first collected in a list using `create_task` -- We won't await them there itself, refer Example 2
- Once all tasks are created, await them using `asyncio.gather()` - all the tasks will be fired together
  - Note: `*list` means each element of list will be passed to the function independently i.e unpacking
- Inside the actual function, we `await` for 1 sec - replicating any IO operation etc; before printing
- **Result: ~1 sec total** (the right way to do it)

```python
import asyncio
import time

async def print_number(number):
    print("Sleeping for 1 second...")
    await asyncio.sleep(1)
    print(f"Number: {number}")


async def main():
    tasks = []
    for i in range(5):
        tasks.append(asyncio.create_task(print_number(i)))
    await asyncio.gather(*tasks)


# Put main() on event control loop
if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    end = time.time()
    print(f"Total time: {end - start} seconds")
```

---

## Example 2: Improper Async Usage - Awaiting Coroutines Directly

Not the best usage of async and await:
- Here, we are awaiting a coroutine (async function), so it just steps into the coroutine and runs it
- Even if there's await inside the coroutine (`asyncio.sleep`), and the control goes to event loop, there's no other thing for event loop to pass and run i.e `print_number(5)` didn't get created when `print_number(1)` is running as we have awaited the coroutine, so it is basically running sequentially!
- **Result: 5 seconds** (same as synchronous)

```python
import asyncio
import time

async def print_number(number):
    print("Sleeping for 1 second...")
    await asyncio.sleep(1)
    print(f"Number: {number}")


async def main():
    for i in range(5):
        await print_number(i)


if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    end = time.time()
    print(f"Total time: {end - start} seconds")
```

---

## Example 3: Bad Async Usage - Missing `await` Inside Coroutine

Bad example usage of async and await - Technically, **a coroutine NEED NOT HAVE an await inside it!** So, not awaiting IO blocking line will cause to run suboptimal.

- Here we are awaiting the `tasks` (which is the right way) i.e we created 5 coroutines and are firing all of them at once
- But the actual async function (`print_number`) **DOESN'T have any await inside it**, so even though all the 5 are fired at once (remember these are NOT run parallely on different resources and require common resources i.e Python is *single threaded*), when one coroutine is running (e.g., `print_number(1)`) and encounters the sleep (real world IO), as we are not awaiting, the control WON'T go back to the event loop and other coroutines won't get a chance to run on the main code thread
- **Result: 5 seconds** (still sequential due to blocking `time.sleep()`)
- Tip: If you have a 3rd party function call, await it to be on safer side.

```python
import asyncio
import time


async def print_number(number):
    print("Sleeping for 1 second...")
    time.sleep(1)  # This is not awaiting, so will literally sleep/block resource.
    print(f"Number: {number}")


async def main():
    tasks = []
    for i in range(5):
        tasks.append(asyncio.create_task(print_number(i)))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    end = time.time()
    print(f"Total time: {end - start} seconds")
```

---

## Example 4: Pure Synchronous Function (Baseline)

Pure synchronous function, nothing async about it - Straightforward.
- **Result: 5 seconds**

```python
import time


def print_number(number):
    print("Sleeping for 1 second...")
    time.sleep(1)
    print(f"Number: {number}")


def main():
    for i in range(5):
        print_number(i)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Total time: {end - start} seconds")
```

---

## Example 5: Syntax Error - `await` Without `async`

Bad example of async and await but because of **syntax**!
- You can't have an `await` without declaring the function as `async`. That will throw an error.

```python
if __name__ == "__main__":
    import asyncio

    # This will throw SyntaxError as we are trying to await without async function
    def wrong_main():
        await asyncio.sleep(1)  # SyntaxError!
        print("Hello, World!")

    asyncio.run(wrong_main())
```

---

## Example 6: Real World - CPU + IO Bound Tasks

Real world example of async (combination of above examples) where we will have both CPU and IO bound tasks.

- We can't speed up the factorial execution itself using async as it's a CPU bound (we can fast it using multithread, caching etc;), but if there are multiple factorials needed, we can run them asynchronously
- Sleep is simulating real world IO bound tasking
- If using `await asyncio.sleep(1)` → the entire code takes **~10 sec** as all the 3 factorial functions run by taking turns
- If using `time.sleep(1)` → it will take **~20 sec**

```python
import asyncio
import time


async def async_factorial(name, n):
    start_time = time.time()
    print(f"Task {name}: Starting factorial({n})")
    result = 1
    for i in range(1, n + 1):
        result *= i
        await asyncio.sleep(1)
        # Yield control back to the event loop (e.g., to run another task)
        # await asyncio.sleep(0)  # Minimal yield for demo, real I/O uses longer sleeps
    print(f"Task {name}: factorial({n}) = {result}")
    print(f"Time taken for Task {name}: {time.time() - start_time:.4f} seconds")
    return result


async def main():
    start_time = time.time()
    # Schedule multiple factorial tasks concurrently
    tasks = [
        asyncio.create_task(async_factorial("A", 10)),
        asyncio.create_task(async_factorial("B", 4)),
        asyncio.create_task(async_factorial("C", 6)),
    ]
    await asyncio.gather(*tasks)
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.4f} seconds")


asyncio.run(main())
```
