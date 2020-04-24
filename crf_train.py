def process_data(x_train,x_valid,x_test,y_train,y_valid,y_test,file_name,p=0.6):
    lines = []
    for i in range(len(x_train)):
        for j in range(len(x_train[i])):
            lines.append(x_train[i][j]+' '+y_train[i][j]+'\n')
        lines.append('\n')
    with open("./data/crf_data/{}_{}.train.crf".format(file_name,str(p)),'w') as f:
        f.write(''.join(lines))
    lines = []    
    for i in range(len(x_test)):
        lines.append('\n'.join(x_test[i])+'\n\n')
    with open("./data/crf_data/{}_{}.test.crf".format(file_name,str(p)),'w') as f:
        f.write(''.join(lines))