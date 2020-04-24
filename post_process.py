import re
price_pattern = r'([0-9零一二三四五六七八九十十百千万亿]+)[块|元]?([0-9零一二三四五六七八九十十百千万亿]*)[角|毛]?'
re_price = re.compile(price_pattern)

num_map = {
    '一': 1,
    '两': 2,
    '二': 2,
    '三': 3,
    '四': 4,
    '五': 5,
    '六': 6,
    '七': 7,
    '八': 8,
    '九': 9,
    '零': 0,
    '十':10,
    '亿':100000000,'万':10000,'千':1000,'百':100
}
# 一百一
num_units = ['亿','万','千','百','十','一']

# side 表示当前子串是左侧还是右侧，左侧为1，右侧为0， 处理 十一这种情况
def str_to_int(num_str):
    if(num_str==''):
        return -1
    re = 0
    for s in num_str:
        if s>='0' and s<='9':
            re = re*10 + int(s)
        else:
            return -1
    return re
        
def convert_num(num_str,pos=0,side=0):
    if num_units[pos] == '一':
        val = str_to_int(num_str)
        # case: 300万
        if val!= -1:
            return val
        # case:一百零三
        num_str = num_str.replace('零','')
        #case:三四万
        if len(num_str)> 1:
            num_str = num_str[0]
        # case : 十一
        if num_str != '':
            return num_map[num_str]
        else:
            return side
    if num_units[pos] not in num_str:
        return convert_num(num_str,pos+1,0)
    split_nums = num_str.split(num_units[pos])
    assert len(split_nums) == 2
    left_num = convert_num(split_nums[0],pos+1,1)
    # case:一千三
    if(len(split_nums[1])==1):
        right_num = num_map[split_nums[1]] * num_map[num_units[pos+1]]
    else:
        right_num = convert_num(split_nums[1],pos+1,0)
    return left_num * num_map[num_units[pos]] + right_num 

def get_entities(text,labels):
    entities = []
    entity = ''
    for i in range(len(text)):
        if entity=='':
            if labels[i] != 'O':
                entity += text[i]
        else:
            if labels[i][0] == 'B' or labels[i] == 'O':
                entities.append(entity)
                entity = '' if labels[i] == 'O' else text[i]
            else:
                entity += text[i]
    return entities

def convert_entity(entities):
    re = []
    for entity in entities:
        prices = ['0','元','0','角']
        parts = re_price.search(entity)
        if parts==None:
            continue
        else:
            parts = parts.groups()
        if parts[0]!='':
            prices[0] = str(convert_num(parts[0]))
        if parts[1]!='':
            prices[2] = str(convert_num(parts[1]))
        re.append(''.join(prices))
    return re
            
def get_prices(text,labels):
    entities = get_entities(text,labels)
    if entities!=[]:
        prices = convert_entity(entities)
    else:
        prices = []
    return prices
    
def postprocess(x_pred,y_pred):
    re = []
    for i in range(len(x_pred)):
        text = x_pred[i]; labels = y_pred[i]
        prices = get_entities(text,labels)
        
