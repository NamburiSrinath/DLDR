######## tldr notes
### We can use async/await only for (almost) independent function calls such as IO, processing on different files etc; If there's dependence, this won't work
### Coroutine - A function which has async key word (mandatory) and await keyword (not mandatory) inside it
### Event Loop - An infinite loop which keeps checking for any async tasks which are awaiting for resources, this is like a scheduler, manager, traffic controller which keeps track of all async tasks and allocates resources to them as and when they are free - Under the hood, has queues to manage tasks
### Do async on tasks (create tasks on coroutines) for best event loop efficiency. async on coroutines is not ideal
### Example 2 - async on coroutines will directly step inside the function and runs/utilizes the resources. So, if there's any sleep (real world IO) wait times inside coroutines, this will block the resources for event loop and thus works same as synchronous
### Example 3 - If there are no `await` inside the coroutines (which is valid), even though all tasks are created and fired at once i.e async on tasks, the resources will still be throttled by one or other coroutine


######### EXAMPLE 1
# Proper example of how async await works
# - main() is kept on event loop control by asyncio.run()
# - all `async` tasks are first collected in a list using create_task -- We won't await them there itself, refer Example 2
# - once all tasks are created, await them using asyncio.gather() - all the tasks will be fired together, Note: *list means each element of list will be passed to the function independently i.e unpacking
# - Inside the actual function, we `await` for 1 sec - replicating any IO operation etc; before printing.
# - As this is right way to do, it took ~1sec to print
#########


########################################################################

# import asyncio
# import time

# async def print_number(number):
#     print("Sleeping for 1 second...")
#     await asyncio.sleep(1)
#     print(f"Number: {number}")


# async def main():
#     tasks = []
#     for i in range(5):
#         tasks.append(asyncio.create_task(print_number(i)))
#     await asyncio.gather(*tasks)


# ### Put main() on event control loop
# if __name__ == "__main__":
#     start = time.time()
#     asyncio.run(main())
#     end = time.time()
#     print(f"Total time: {end - start} seconds")

########################################################################

######### EXAMPLE 2
# Bad example usage of async and await
# - Here, we are awaiting a coroutine (async function), so it just steps into the coroutine and runs it
# - Even if there's await inside the coroutine (sleep), and the control goes to event loop, there's no other thing for it to run i.e print_number(5) didn't get created when print_number(1) is running as we have awaited the coroutine, so it will run
# - So, this will take 5 sec
#########


########################################################################

# import asyncio
# import time

# async def print_number(number):
#     print("Sleeping for 1 second...")
#     await asyncio.sleep(1)
#     print(f"Number: {number}")


# async def main():
#     for i in range(5):
#         await print_number(i)


# if __name__ == "__main__":
#     start = time.time()
#     asyncio.run(main())
#     end = time.time()
#     print(f"Total time: {end - start} seconds")
########################################################################


######### EXAMPLE 3
# Bad example usage of async and await - A co
# - Here we are awaiting the tasks (which is the right way) i.e we created 5 coroutines and are firing all of them at once
# - But the actual async function (print_number) DOESN'T have any await inside it, so even though all the 5 are fired at once (remember these are NOT run parallely on different resources and require common resources), when one coroutine is running (eg: print_number(1)) and encounter the sleep (real world IO), as we are not awaiting, the control WON'T go back to the event loop and other coroutines won't get a chance to run
# - So, this will also take 5 seconds
#########

########################################################################
# import asyncio
# import time


# async def print_number(number):
#     print("Sleeping for 1 second...")
#     time.sleep(1)
#     print(f"Number: {number}")


# async def main():
#     tasks = []
#     for i in range(5):
#         tasks.append(asyncio.create_task(print_number(i)))
#     await asyncio.gather(*tasks)


# if __name__ == "__main__":
#     start = time.time()
#     asyncio.run(main())
#     end = time.time()
#     print(f"Total time: {end - start} seconds")
########################################################################

######### EXAMPLE 4
# Pure synchronous function, nothing async about it - Straightforward, takes 5 seconds
#########

########################################################################
# import time


# def print_number(number):
#     print("Sleeping for 1 second...")
#     time.sleep(1)
#     print(f"Number: {number}")


# def main():
#     for i in range(5):
#         print_number(i)


# if __name__ == "__main__":
#     start = time.time()
#     main()
#     end = time.time()
#     print(f"Total time: {end - start} seconds")
########################################################################

######### EXAMPLE 5
# Bad example of async and await but more on syntax!!
# - You can't have an await without declaring the function as async. That will throw an error
#########

########################################################################
# if __name__ == "__main__":
#     import asyncio

#     # This will throw SyntaxError as we are trying to await without async function
#     def wrong_main():
#         await asyncio.sleep(1)
#         print("Hello, World!")

#     asyncio.run(wrong_main())
########################################################################
