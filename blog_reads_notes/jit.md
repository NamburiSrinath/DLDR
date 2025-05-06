<b> Blog link: </b> https://kipp.ly/jits-intro/

---
- When we run a program, it is either interpreted or compiled in someway!
- Compiler/interpreter can also be known as "implementation" of a language and one language can have multiple implementations. 
- Interpreter is a program that **directly executes** the code. For eg: Ruby, Python, PHP are written in C. If we see this example, interpret basically takes in code and according to what is there in the code does that thing!!
```
func interpret(code string) {
  if code == "print('Hello, World!')" {
    print("Hello, World");
  } else if code == “x = 0; x += 4; print(x)” {
    variable_x := 0
    variable_x += 4
    print(x)
  }
}
```
- Compiler is a program that **translates code** from some language to a different language (usually a low-level). C, Go and Rust are compiled languages!. This basically takes in code, gets the compiled version of code and that is given to a different function to execute!!
```
func compile(code string) {
  []byte compiled_code = get_machine_code(code);
  write_to_executable(compiled_code); // someone else will execute me later
}
```
- Python is an interpreted language while C is a compiled language i.e C outputs a machine code file that can be natively understood by computer -> compile and run steps are fully distinct!
![alt text](image.png)
- Java -> 2 steps
  - Java code to Bytecode IR (Intermediate representation)
  - Bytecode IR -> JIT compiled (this involves interpretation)
- Python source code implementation is also compiled to bytecode (`.pyc`) and this bytecode is then interpreted by a virtual machine. Which means virtual machines has their interpreters in bytecode because:
  - We care less about the compile time (translation from one language to another language)
  - We care more about the interpretation/logic of the code to be executed. So, to be more efficient as possible to interpret, using bytecode language as an interpreter is better!
- Also, having bytecode is how languages check syntax before execution! For example, in the below code, if we don't check syntax before runtime, we will waste 1000 seconds before hitting the syntax error!
```
sleep(1000)
bad syntax beep boop beep boop
```
- Interpreted languages are slower than compiled languages because of the dynamic-ness/flexibility present in the interpreted languages which will introduce overheads like which function to call next, how to pass the data etc; 
- JIT (Just-in-Time) doesn't compile code AOT (Ahead-of-Time) but still compiles source to machine code thus is NOT an interpreter! Basically, JITs compile code at runtime while your program is still running!!
```
func jit_compile(functions []string) {
  for function := range functions {
    compiled_fn := get_machine_code(function);
    compiled_fn.execute();
  }
}
```
- JIT compiling C will make it slower as we are adding compiling time to the execution time while JIT compiling Python will be faster as it compil;ation + execution of machine code is faster than interpreting! So, JIT means -- compiles code right before the code needs to be used -- "just in time"
- In AOT (Ahead of Time) compilation, everything has to be compiled beforehand whether those are useful or not!! That's the overhead!
- JIT - Lazy AOT compiler!
- Compiling a code just in time might not produce an optimized code as the compiler might not have all the information needed for the best optimal machine code. If you try to get your code to compile sooner, less data will be available, the compiled code will not be as efficient and peak performance will be lower.
- Read mode about interpretation, profiling, compiling, warming up etc; as this blog is a bit dense!!!