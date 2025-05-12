#!/usr/bin/env python
# coding: utf-8

# In[5]:


def convolution(input_tensor, kernel, bias, stride):
    batch, h, w, c_in = input_tensor.shape
    kh, kw, _, c_out = kernel.shape
    
    output_h = (h - kh) // stride + 1
    output_w = (w - kw) // stride + 1
    
    output_tensor = np.zeros((batch, output_h, output_w, c_out))
    
    for b in range(batch):
        for i in range(0, h - kh + 1, stride):
            for j in range(0, w - kw + 1, stride):
                patch = input_tensor[b, i:i+kh, j:j+kw, :]
                conv = np.sum(patch[:, :, :, np.newaxis] * kernel, axis=(0, 1, 2)) + bias
                output_tensor[b, i//stride, j//stride, :] = conv
                
    return output_tensor
input_tensor = np.load('tensor.npy')
kernel = np.load('kernel.npy')
bias = np.load('bias.npy')

with open('task.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        stride = int(row['stride'])

result = convolution(input_tensor, kernel, bias, stride)

np.save('seminar03_conv.npy', result, allow_pickle=False)


# In[ ]:




