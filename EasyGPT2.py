import sys
import json
import struct
import numpy as np

#1. First get prompt from command line
prompt = sys.argv[1]
print("Input: " + prompt)

#2. Split by sequences of spaces/symbols/alpha. Retain prefix space for sequences of alpha or symbols. Alpha sequences can start with '
# she'll abcd1234 abcd ##$%     abc    123 => ["she", "'ll", "abcd", "1234", " abcd", "##$%", "    ", " abc", "   ", " 123"]
tokens = []
pos = 0
word = ""
wordType = -1 #-1 for spaces, 0 for symbols, 1 for alpha

for c in prompt:
    if c == " ":
        if 0 <= wordType:
            tokens.append(word)
            word = ""
        wordType = -1
    else:
        if wordType < 0 and 1 < len(word): #many spaces
            tokens.append(word[:-1]) # Keep all characters except last
            word = " "
        elif wordType == 1 - c.isalpha() and word!="'": #different type and not starting with '
            tokens.append(word)
            word = ""
        wordType = c.isalpha()
    word = word + c
tokens.append(word)

#3. Create byte to character mapping for readability and debugging ease
mapping = {}
for i in range(0, 256):
    mapping[i] = chr(i)
    if (i <= 32) or (127 <= i and i <= 160) or (i == 173) :
            mapping[i] = chr(256 + i)

#4. Convert possibly unicode text to UTF-8 bytes and map each byte to readable characters
for i in range(0, len(tokens)):
    t = tokens[i].encode("utf-8")
    encoded = ""
    for b in t:
        encoded = encoded + mapping[b]
    tokens[i] = encoded

#5. Prepare vocab file for BPE
vocabFile = open("data\\vocab.bpe", "r", encoding="utf-8")
vocabData = vocabFile.read() #read whole file
vocabLines = vocabData.split("\n") #get all the individual lines
vocabUseful = vocabLines[1:-1] #Do not keep first line that contains #version: 0.2 and last empty line
vocabDict = {}
for i in range(0, len(vocabUseful)):
    res = vocabUseful[i].split(" ")
    if res[0] not in vocabDict:
            vocabDict[res[0]] = {}
    vocabDict[res[0]][res[1]] = i

#6. BPE merging using vocab file order as priority
MergedTokens = [] 
for t in tokens:
    l = [b for b in t] #convert token in array of characters
    while True :
        minRank = -1
        mergePos = -1
        for i in range(0, len(l) - 1):
            if l[i] in vocabDict and l[i + 1] in vocabDict[l[i]]:
                rank = vocabDict[l[i]][l[i + 1]]
                if rank < minRank or mergePos == -1: # find the merge opportunity with lowest rank
                    mergePos = i
                    minRank = rank

        if mergePos == -1 :
            break  # did not find any merge opportunities so we are done

        l2 = l[: mergePos + 1] + l[mergePos + 2 :]
        l2[mergePos] = l2[mergePos] + l[mergePos + 1]
        l = l2

    for p in l :
        MergedTokens.append(p)

#7. Convert token into numerical values 50257 values in vocabulary
encoderFile = open("data\\encoder.json", "r")
encoderJson = json.load(encoderFile)
Values = []
for t in MergedTokens:
    Values.append(encoderJson[t])
print( "Tokens: " + str( Values ) )

#8. Read model parameters
ParamsJson = json.load(open("data\\hparams.json", "r")) # n_vocab=50257, n_ctx=1024, n_embd=768, n_head=12, n_layer=12
n_layer = ParamsJson["n_layer"]
n_head = ParamsJson["n_head"]

#9. Load coefficients
def LoadTensor( file_name ): # Custom naive binary tensor format
    inputFile = open(f"data\\{file_name}.tensorbin", "rb") # open file as binary
    dims = int.from_bytes(inputFile.read(4), byteorder='little')  # read number of dimensions as the first 4 bytes integer in the file
    dimensions = [int.from_bytes(inputFile.read(4), byteorder='little') for d in range(0, dims)] # create an array with each dimension read from the file as 4 byte integer
    num_elements = np.prod( dimensions ) # np.prod returns the product of each component in the array
    return np.frombuffer( inputFile.read(), dtype=np.float32).reshape( dimensions ) # read all the file data left, convert it to linear array of floats, and the shape with tensor dimensions

print( "Loading tensors" )
wte = LoadTensor("wte") #Tokens Embeddings 50257 possible vectors of size 768
wpe = LoadTensor("wpe") #Positional Embeddings 1024 possible vectors of size 768

#10. Convert tokens into embeddings 768 columns, number of rows is the number of tokens, each row is the sum of token and positional embeddings
Embeddings=np.empty( [ len(Values), wpe.shape[1] ] )
for i in range(0, len(Values)):
    Embeddings[i]=np.add(wte[Values[i]], wpe[i]) # add vectors component by component for each j res[i][j] = wte[Values[i]][j] + wpe[i][j]

#11. Load transformer model
attn_c_attn_b = [[[],[],[]] for i in range(0, n_layer)]
attn_c_attn_w = [[[],[],[]] for i in range(0, n_layer)]
for i in range(0, n_layer): # model has n_layer layers
    qkv_names = ["q","k","v"]
    for j in range(0, 3): # each layer has 3 pieces q, k, v
        attn_c_attn_b[i][j] = [LoadTensor(f"layer{i}_attn_c_attn_b_{qkv_names[j]}_h{h}") for h in range(0, n_head)] # array of n_head vectors of dimension n_embd/n_head
        attn_c_attn_w[i][j] = [LoadTensor(f"layer{i}_attn_c_attn_w_{qkv_names[j]}_h{h}") for h in range(0, n_head)] # array of n_head matrices of dimension n_embd x n_embd/n_head
attn_c_proj_b = [LoadTensor("layer%d_attn_c_proj_b" % i) for i in range(0, n_layer)] # array of n_layer vectors of dimension n_embd
attn_c_proj_w = [LoadTensor("layer%d_attn_c_proj_w" % i) for i in range(0, n_layer)] # array of n_layer matrices of dimension n_embd x n_embd
ln_1_b = [LoadTensor("layer%d_ln_1_b" % i) for i in range(0, n_layer)] # array of n_layer vectors of dimension n_embd
ln_1_g = [LoadTensor("layer%d_ln_1_g" % i) for i in range(0, n_layer)] # array of n_layer vectors of dimension n_embd
ln_2_b = [LoadTensor("layer%d_ln_2_b" % i) for i in range(0, n_layer)] # array of n_layer vectors of dimension n_embd
ln_2_g = [LoadTensor("layer%d_ln_2_g" % i) for i in range(0, n_layer)] # array of n_layer vectors of dimension n_embd
mlp_c_fc_b = [LoadTensor("layer%d_mlp_c_fc_b" % i) for i in range(0, n_layer)] # array of n_layer vectors of dimension 4*n_embd
mlp_c_fc_w = [LoadTensor("layer%d_mlp_c_fc_w" % i) for i in range(0, n_layer)] # array of n_layer matrices of dimension n_embd x 4*n_embd
mlp_c_proj_b = [LoadTensor("layer%d_mlp_c_proj_b" % i) for i in range(0, n_layer)] # array of n_layer vectors of dimension n_embd
mlp_c_proj_w = [LoadTensor("layer%d_mlp_c_proj_w" % i) for i in range(0, n_layer)] # array of n_layer vectors of dimension 4*n_embd x n_embd
ln_f_b = LoadTensor("ln_f_b") # vector of dimension n_embd
ln_f_g = LoadTensor("ln_f_g") # vector of dimension n_embd

#12. Apply Neural Network aka Buzz Word Transformer
def normalize_scale_shift(x, g, b, eps: float = 1e-5): # x dim nb_tokens x n_embd, g dim n_embd, b dim n_embd
    res = np.empty(x.shape)
    for r in range(0, x.shape[0]): # normalize then scale and shift each row individually
        mean = np.sum(x[r]) / x.shape[1] # average of the row r by summing all elements of the vector and divide by number of values
        sum = 0.0
        for j in range(0, x.shape[1]): # calculate variance of row r by summing square of difference with mean for every vector component
            delta = x[r][j] - mean
            sum = sum + delta * delta
        stdDev = np.sqrt(sum / (x.shape[1] - 1)) + eps # reduce risk of division by 0
        for j in range(0, x.shape[1]):
            res[r][j] = g[j]*(x[r][j]-mean)/stdDev+b[j]
    return res

for outputTokens in range( 0, 8 ):
    x = np.array( Embeddings ) # input sequence number of [n_seq, n_embd]

    for layer in range(0, n_layer):
        #print("Layer %d" % layer, end="\r")
        # multi-head causal self attention
        xn = normalize_scale_shift(x,ln_1_g[layer],ln_1_b[layer])
        temp = np.empty(xn.shape) #concatenated result of all heads
        for head in range(0, n_head):
            # qkv projection
            q = np.matmul(xn, attn_c_attn_w[layer][0][head] ) + attn_c_attn_b[layer][0][head] # [n_seq, n_embd/n_head] -> [n_seq, n_embd/n_head]
            k = np.matmul(xn, attn_c_attn_w[layer][1][head] ) + attn_c_attn_b[layer][1][head] # [n_seq, n_embd/n_head] -> [n_seq, n_embd/n_head]
            v = np.matmul(xn, attn_c_attn_w[layer][2][head] ) + attn_c_attn_b[layer][2][head] # [n_seq, n_embd/n_head] -> [n_seq, n_embd/n_head]
            qkt = np.matmul( q, k.T ) * ( 1 / np.sqrt(q.shape[-1]) ) # [n_seq, n_seq]
            # softmax and mask
            for r in range(0, qkt.shape[0]): # apply softmax to each masked row of qkt
                maxRow = max(qkt[r][:r+1]) # find the max value of the first r elements of the row, the rest is masked
                sum = 0
                for c in range(0, r+1):
                    qkt[r][c]=np.exp(qkt[r][c] - maxRow) # apply soft mask formula first part to first r elements of the row
                    sum = sum + qkt[r][c]
                for c in range(0, r+1):
                    qkt[r][c]=qkt[r][c]/sum  # apply soft mask formula second part to first r elements of the row
                for c in range(r+1, qkt.shape[1]):
                    qkt[r][c]=0.0 # mask the rest of the row
            attention = np.matmul( qkt, v ) # for this head attention is q.Transpose(k).v [n_seq, n_embd/n_head]
            for j in range(0, xn.shape[0]): # concatenate attention heads in temp
                for k in range(0, attention.shape[1]):
                    temp[j][k+head*attention.shape[1]]=attention[j][k]
        temp = np.matmul(temp, attn_c_proj_w[layer] ) + attn_c_proj_b[layer] # [n_seq, n_embd] -> [n_seq, n_embd]
        x = x + temp
        
        # position-wise feed forward network
        xn = normalize_scale_shift(x,ln_2_g[layer],ln_2_b[layer])
        temp = np.matmul(xn, mlp_c_fc_w[layer]) + mlp_c_fc_b[layer] # widen / [n_seq, n_embd] -> [n_seq, 4*n_embd]
        for i in range(0, temp.shape[0]): # independent gelu per component https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html
            for j in range(0, temp.shape[1]):
                temp[i][j] = 0.5 * temp[i][j] * (1 + np.tanh(np.sqrt(2 / np.pi) * (temp[i][j] + 0.044715 * temp[i][j]*temp[i][j]*temp[i][j])))
        temp = np.matmul(temp, mlp_c_proj_w[layer] ) + mlp_c_proj_b[layer] # shrink / [n_seq, 4*n_embd] -> [n_seq, n_embd]
        x = x + temp
     
    x = normalize_scale_shift(x,ln_f_g,ln_f_b) # only apply to last row since that is the only one we need
    
    #13. REVERSE OF STEP 9 CONVERT EMBEDDING INTO TOKEN BY FINDING THE VALUE WITH MAXIMUM MATCH
    output = np.matmul(x[-1], wte.T) #LAST NEW EMBEDDING DOT PRODUCT WITH ALL EMBEDDINGS TO FIND BEST FIT
    next_value = 0
    for i in range(1, output.shape[0] ): # FINDING INDEX OF MAXIMUM VALUE EQUIVALENT OF np.argmax
        if output[next_value] < output[i]:
            next_value = i

    #14. Update Embeddings
    Values.append(next_value)
    print( f"Generated tokens: {Values[:]}", end="\r" )
    next_embedding = np.add(wte[next_value], wpe[len(Values)])
    Embeddings = np.vstack( [ Embeddings, next_embedding ] ) # add new embedding as row at the end of previous embeddings 
    
#15. CREATE DICTIONNARY BY INT TO REVERSE OF STEP 7
decoder = {}
for k in encoderJson:
    v = encoderJson[k]
    decoder[v] = k

#16. REVERSE OF STEP 7 DECODE NUMBER
decoded = []
for v in Values:
    decoded.append( decoder[v] )

#17. CREATE INVERSE TABLE OF STEP 3
reversemapping = {}
for i in range(0, 256):
    reversemapping[mapping[i]] = i

#18. REVERSE OF STEP 4
inverseDecoded = []
for token in decoded:
    for b in token:
        inverseDecoded.append(reversemapping[b])
inverseDecoded = bytearray(inverseDecoded).decode("utf-8")
print( f"Output: {inverseDecoded}" )
