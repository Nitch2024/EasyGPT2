# EasyGPT2
Modest python rewrite of [openai/gpt-2](https://github.com/openai/gpt-2) inspired by the awesome [karpathy/minGPT](https://github.com/karpathy/mingpt) and [jaymody/picoGPT](https://github.com/jaymody/picoGPT) but biased toward education.

1. This project has minimal dependencies (sys, json, struct and numpy). Tensorflow is not needed, so no need to deal with Conda or tensorflow lack of support for latest python versions.
2. Only one source file to make things linear and not be confused by code constructs or jumping through different files and functions. 
3. The code is not optimized for speed, size or any desirable engineering attribute. The code is meant to be read and understood easily. Numpy is used since naive custom matrix functions are too slow.
4. The code does explain why the algorithm or attention mechanism works. This is not a "why" tutorial, it is a "how" tutorial. The code shows each detailed step with less room for interpretation than a paper or diagram.
5. The trained data from [OpenAI original trained GPT2 data (gpt2-small 124M)](https://openaipublic.blob.core.windows.net/gpt-2/models) was pre-converted to a naive tensor format to eliminate the tensorflow dependency.
6. This tutorial has nothing about training the model. It just shows how to use the pre-trained model and apply it.

If you have any question, need any help or have any suggestion to improve this tutorial email me a cedrickgithub+easygpt2 (at) gmail.com

#### Setup

1. Download easygpt2.py and the converted [gpt2 124M tensors](https://drive.google.com/file/d/1kfqzwWfSG7h6eoC9JIvmOywFj9kzb5_D/view?usp=drive_link)
2. Decompress all the files from converted.7z into a folder called converted sitting next to easygpt2.py
   
#### Usage in command line
```
python easygpt2.py "Alan Turing theorized that computers would one day become"
```

#### Result shown in the command line window
```
Input: Alan Turing theorized that computers would one day become
Tokens: [36235, 39141, 18765, 1143, 326, 9061, 561, 530, 1110, 1716]
Loading tensors
Generated tokens: [36235, 39141, 18765, 1143, 326, 9061, 561, 530, 1110, 1716, 262, 749, 3665, 8217, 319, 262, 5440, 13]
Output: Alan Turing theorized that computers would one day become the most powerful machines on the planet.
```

#### Overview

The code is broken down in 18 logical steps.
* Steps 1 to 7 transform the input text into tokens
* Steps 8 to 11 convert tokens into embeddings and load tensors for the model
* Steps 12 to 14 execute the model and decode a new token in a loop for the number of desired tokens.
* Steps 15 to 18 transform the new tokens into words



