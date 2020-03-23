def labelParser(path='./Model_config/label_map.pbtxt'):
    labelFile = open(path,'r')
    labelMap = {}
    idn = int()
    display_name = str()
    for line in labelFile.readlines():
        if line.replace(' ','')[0:2] == 'id':
            idn = line.replace('id: ','').replace('\n','').replace(' ','')
        elif line.replace(' ','')[0:12] == 'display_name':
            try:
                labelMap[idn] = line[17:-2] 
            except:
                pass
    return labelMap


