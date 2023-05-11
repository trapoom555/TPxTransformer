# Transformer
This project aims to crack [Attention is all you need](https://arxiv.org/abs/1706.03762) paper and write it in `PyTorch 🔥`.<br>
"Transformer" model is proposed in this paper and has revolutionized Seq2Seq task. It performs better than earlier state-of-the-art (SOTA) RNNs 
in terms of training time and accuracy.  
  
  
The reasons for that are
1. Transformer considers all input Seq at the same time. There isn't *Information Bottleneck* issue as RNN that sequentially process the input.
2. The design of Data in / Data out and *Causal Mask Attention* makes the training process parallelizable.

## Architecture of Transformer

<img width="539" alt="image" src="https://github.com/trapoom555/TPxTransformer/assets/36850068/ac47f25c-c2e5-4e16-bf3d-0a7d594017b9">. 

Shortly, the model consists of two building blocks
1. Encoder : Encoder constructs embedded feature vector of the input sequence.
2. Decoder : Decoder combines embedded feature vector from encoder and previous outputs to predict next token (output)

You can read more details about Transformer's Architecture on my Blog soon!!  

## Test Data (Mock)

I've mock the data to train and validate *Seq2Seq* behavior of Transformer model that I've written.  
For that, an output data has to be sequential-dependent on sequence of input data.  

So, `Reverse Sequence, Inverse Sum` dataset is used as a mock data.  
Here's an example for generating the data. If we have input as
```
x: [3, 9, 5, 6]
```
We reverse order first and do Inverse Sum. Let's say the number need to sum up to `17`
```
reversed order  : [6, 5, 9, 3]
inverse sum     : [11, 12, 8, 14]
```
Finally we have will the data like this
```
x: [3, 9, 5, 6]
y: [11, 12, 8, 14]
```
In this experiment, I preserved token `1` to be a start token and `2` to be a stop token and `0` for padding. 
After adding start and stop token the data should look like this
```
x: [1, 3, 9, 5, 6, 2]
y: [1, 11, 12, 8, 14, 2]
```
After shifting and modifing data according to [Attention is all you need](https://arxiv.org/abs/1706.03762), this is what the final data looks like. 
```
Encoder input:  [1, 3, 9, 5, 6, 2]
Decoder input:  [1, 11, 12, 8, 14]
Decoder output: [11, 12, 8, 14, 2]
```
## The Result

Our model can achieve `99.99%` on this datasets by training just a few minutes on *NVIDIA 1050* GPU !
