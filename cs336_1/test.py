import torch

char_en = "A"

char_cn = "你"
print(char_en, ord(char_en))
print(char_cn, ord(char_cn))
print(char_cn, hex(ord(char_cn)))

text = "牛"
encode_type = text.encode("utf-8")

print(text)
print(encode_type)
print(ord(text))
print("字节列表", [bin(b) for b in encode_type])

# 将20320编码为你
# 29275 牛
code_point = 29275
print(code_point)
print(hex(code_point))
print(bin(code_point))
# pattern 1110 xxxx | 10 xxxxxx | 10 xxxxxx
# 匹配 0111 0010 0101 1011
# 结果 ['1110 0111', '10 001001', '10 011011']



s = 5
batch_dims = (2, 3)
pos = torch.arange(s, device='cuda').expand(*batch_dims, s)
print(pos.shape)   # torch.Size([2, 3, 5])
print(pos[0,0])    # tensor([0,1,2,3,4])

print(pos)

print("连续性判断：连续，表示内存中是行优先存储；不连续，一般是通过转置、切片操作得到，内存中不连续")

# 连续张量
x = torch.arange(6)         # shape (6,)
y = x.view(2, 3)            # OK，返回 view，y 与 x 共享内存
print("y.is_contiguous()", y.is_contiguous())   # True

# 转置通常产生非连续张量
xt = y.t()                  # shape (3,2)，通常 non-contiguous
print("xt.is_contiguous()", xt.is_contiguous()) # False

# view 在非连续张量上会报错
try:
    z = xt.view(6)
except Exception as e:
    print("view error:", e)

# reshape 会工作（有时返回 view，有时返回 copy）
zr = xt.reshape(6)
print("reshape result contiguous?", zr.is_contiguous())


from nn import softmax

x = torch.arange(6)
print(x)
y = softmax(x, dim=-1)
print(y)

x = x.float()
y = torch.softmax(x, dim=-1)
print(y)