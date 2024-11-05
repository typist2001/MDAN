import re
input='''

'''
lines=input.split('\n')
res=""
for i in lines:
    if "Best" in i:
        pattern = re.compile('.*psnr:|.*ssim:|Best:.*')
        res+=pattern.sub("",i ).strip()+"\t"
res=res.strip()
print(res)
