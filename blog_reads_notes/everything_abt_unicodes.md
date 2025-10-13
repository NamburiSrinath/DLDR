<b> Blog: </b> https://www.joelonsoftware.com/2003/10/08/the-absolute-minimum-every-software-developer-absolutely-positively-must-know-about-unicode-and-character-sets-no-excuses/

---
- Writing code for international software requirements: Need to understand about unicodes etc; It's beyond ASCII
- Plain text == ascii == characters are 8 bits --> totally wrong!
- ASCII: 128 characters represented using 7 bits. 32 - 127 are the ones that are printable. 0-31 are actionable character representations/control characters i.e 7 is for beeping computer.
- 128 - 255 are like free characters so every country started using it differently. Thus one country's document was unreliably translated when looking in another country as 128-255 mapping is different for each of them i.e there's no standardization.
- So, people started using different codepages (CPs) which is basically the key-value mapping. As long as your computer has the CP installed, we can understand the other document. But having multiple codepages at same time is still not possible.
- So, Unicode has been invented because of internet as characters started moving across nations!
- MYTH: Unicode: 16-bit code where each character takes 16 bits, so there are 65,536 possible characters.   
- ASCII or ANSI: Letter maps to some bits which we store on disk or memory. Unicode: Letter maps to *code point*, which is a magic number by some agreement/consortium.
- There's no theoretical upper limit on number of characters that can be represented by Unicode.
- Hello: U+0048 U+0065 U+006C U+006C U+006F (in unicode). Just a bunch of numbers! How to store this in memory? -- *Encodings*
- The earliest encodings is basically -- Store everything using 2 bytes (thus the myth!). This is UCS-2 method or UTF-16 as it has 16 bits. So, 00 48 or 48 00 both were used to represent `H`, high-endian or low-endian. So, folks has to come up with adding "Unicode byte order mark" FE FF which describes whether one is following high-endian or low-endian.
- For english only folks, unicode was not worth it as their characters doesn't use a lot of those bytes thus their memory requirements increased a lot!
- UTF-8 came along which is another system to store your string of unicode code points in memory using *8 bit bytes*!! Every codepoint from 0-127 is stored in a *single byte*. Only code points above 128 are stored using 2, 3 upto 6 bytes. This way, most used characters are represented using 0-127 thus using only 1 byte which reduces the memory requirements. Also, another neat side effect that English text looks exactly the same in UTF-8 as it did in ASCII, so Americans don’t even notice anything wrong.
- UCS-4 which is basically UTF-32 uses 32 bits to store and that too in a uniform way i.e every character gets 32 bits!! That's a lot of memory wastage.
- If a character is not found in an encoding scheme, it returns a `?`
- *It does not make sense to have a string without knowing what encoding it uses. There Ain’t No Such Thing As Plain Text.* If you have a string, in memory, in a file, or in an email message, you have to know what encoding it is in or you cannot interpret it or display it to users correctly.
- HTML web pages should contain the `meta` and `charset` right after <head> to determine the encoding that has been used so it can render accordingly. In the event that the web author forgot to add it, the internet looks at histogram of the characters that has been used in that web, and because every human language has a different characteristic histogram of letter usage, looks at common encodings and tries to decode which has a good chance of working! If <meta charset="UTF-8"> is there at the start of a webpage, it's good, the system can decode accordingly using UTF-8 mapping. 